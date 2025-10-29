from project.utils.logger import setup_logger
import torch
import warnings
import os
from pyspark.sql import SparkSession
import pyarrow.fs
import torch.nn as nn

from project.data.data_preprocessor import prepare_data_partitions
from project.data.data_loader import create_pytorch_dataloader
from project.models import create_model
from project.utils.hdfs_utils import save_and_upload_report
from project.utils.visualization import plot_and_save_confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_loop(model, dataloader, criterion, device):
    """Evaluates the model on a dataloader that wraps an IterableDataset."""
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    all_preds, all_labels = [], []
    
    # Suppress the UserWarning from PyTorch about the length of the IterableDataset
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Length of IterableDataset .* was reported to be .* but .* samples have been fetched.",
            category=UserWarning
        )
        with torch.no_grad():
            for features, labels, weights in dataloader:
                features, labels = features.to(device), labels.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                
                total_loss += loss.item() * features.size(0)
                total_correct += (predicted == labels).sum().item()
                total_samples += features.size(0)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
    if total_samples == 0:
        return 0.0, 0.0, [], []
        
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    
    return avg_loss, accuracy, all_labels, all_preds

def evaluate_on_test_set(training_result, args):
    """Loads the best model and evaluates it on the test set."""

    driver_log_file = args["logger_file"]
    logger = setup_logger(rank="DRIVER", log_file=driver_log_file)
    
    logger.info(f"\nStarting FINAL evaluation on TEST set...")

    # Extract correct paths
    data_paths = args["data_source_paths"]
    artifact_paths = args["artifact_storage_paths"]
    output_dir = artifact_paths["output_dir"]  # HDFS path
    test_data_path = data_paths["test_path"]  # Alluxio path
    temp_dir_base = data_paths.get("temp_dir", "/tmp")
    
    try:
        # 1. Load model from HDFS
        model_path_hdfs = os.path.join(
            output_dir, f"best_{args['model_type']}_model.pth"
        )
        logger.info(f"Loading BEST model from artifact storage: {model_path_hdfs}")
        model = create_model(**args)

        fs, hdfs_model_path = pyarrow.fs.FileSystem.from_uri(model_path_hdfs)
        with fs.open_input_file(hdfs_model_path) as f:
            model.load_state_dict(torch.load(f, weights_only=True))

        model.to(torch.device("cpu"))
        logger.info("Model loaded successfully.")

        # 2. Prepare test data from Alluxio
        timestamp = output_dir.split("_")[-1]
        test_temp_dir = f"{temp_dir_base}/test_data_{args['model_type']}_{timestamp}"
        test_paths, test_samples = prepare_data_partitions(
            SparkSession.getActiveSession(), test_data_path, 1, test_temp_dir
        )

        # Add a small delay to allow Alluxio to fully commit the data for short-circuit reads
        logger.info("Waiting for 3 seconds to allow data to settle in Alluxio...")
        import time
        time.sleep(3)

        test_dataloader = create_pytorch_dataloader(
            test_paths, args["batch_size"] * 2, test_samples, is_training=False
        )

        if test_dataloader:
            # 3. Evaluate
            _, test_acc, all_labels, all_preds = evaluate_loop(
                model, test_dataloader, nn.CrossEntropyLoss(), torch.device("cpu")
            )
            logger.info(f"--- FINAL TEST SET PERFORMANCE ({args['model_type'].upper()}) ---")
            logger.info(f"  Test Accuracy: {test_acc:.4f}")

            # 4. Save reports to HDFS
            report = classification_report(
                all_labels,
                all_preds,
                target_names=args["class_names"],
                output_dict=True,
            )
            report["model_info"] = {**training_result, "test_accuracy": test_acc}
            save_and_upload_report(report, f"final_test_report.json", output_dir)

            cm = confusion_matrix(all_labels, all_preds)
            plot_and_save_confusion_matrix(
                cm, args["class_names"], output_dir, args["model_type"]
            )

    except Exception as e:
        import traceback

        logger.info(f"[ERROR] Could not perform final evaluation: {e}")
        traceback.logger.info_exc()