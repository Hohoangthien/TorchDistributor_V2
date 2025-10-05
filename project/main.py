"""
Main entry point for the training pipeline.
"""

import os
import torch
import torch.nn as nn
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexerModel
from pyspark.ml.torch.distributor import TorchDistributor
from datetime import datetime
import tempfile
import pyarrow.fs
from urllib.parse import urlparse

from project.data.spark_utils import init_spark
from project.utils.config import load_config, parse_cli_args
from project.data.data_preprocessor import prepare_data_partitions
from project.data.data_loader import create_pytorch_dataloader
from project.training.trainer import training_function
from project.training.evaluator import evaluate_loop
from project.models import create_model
from project.utils.hdfs_utils import save_and_upload_report, delete_hdfs_directory, upload_log_file
from project.utils.visualization import plot_and_save_confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
from project.utils.logger import setup_logger


# Initialize logger for the driver
driver_log_file = "/tmp/driver.log"
logger = setup_logger(rank="DRIVER", log_file=driver_log_file)

def main():
    cli_args = parse_cli_args()
    config = load_config(cli_args.config)

    spark_config = config["spark"]
    data_source_config = config["data_source"]
    artifact_storage_config = config["artifact_storage"]
    training_config = config["training"]
    model_params_config = config["model_params"]

    spark = init_spark(spark_config["deploy_mode"])

    logger.info(f"\nStarting {config['project_name']}")
    model_type = training_config["model_type"]
    logger.info(f"Selected Model: {model_type.upper()}")

    # --- Prepare model parameters ---
    final_model_params = model_params_config.get("default", {})
    if model_type in model_params_config:
        final_model_params.update(model_params_config[model_type])

    # --- Prepare a single dictionary of arguments for the TorchDistributor ---
    # This dictionary contains all information needed by the remote workers.
    distributor_args = {
        "data_source_paths": data_source_config,  # Paths for reading data (Alluxio)
        "artifact_storage_paths": artifact_storage_config,  # Paths for writing results (HDFS)
        **training_config,
        **final_model_params,
    }

    try:
        label_indexer_model = StringIndexerModel.load(
            data_source_config["label_indexer_path"]
        )
        distributor_args["num_classes"] = len(label_indexer_model.labels)
        distributor_args["class_names"] = label_indexer_model.labels
        first_row = (
            spark.read.parquet(data_source_config["train_path"])
            .select("scaled_features")
            .first()
        )
        distributor_args["num_features"] = first_row["scaled_features"].size
    except Exception as e:
        logger.info(f"Error reading metadata: {e}. Using defaults.")
        (
            distributor_args["num_classes"],
            distributor_args["class_names"],
            distributor_args["num_features"],
        ) = (2, ["0", "1"], 39)

    # --- Prepare data partitions on temp storage (Alluxio) ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir_base = data_source_config.get("temp_dir", "/tmp")

    val_path = os.path.join(
        os.path.dirname(data_source_config["train_path"]), "val_df.parquet"
    )
    val_temp_dir = f"{temp_dir_base}/val_data_{model_type}_{timestamp}"
    val_paths, _ = prepare_data_partitions(spark, val_path, 1, val_temp_dir)
    distributor_args["validation_file_path"] = val_paths[0] if val_paths else None

    train_temp_dir = f"{temp_dir_base}/train_data_{model_type}_{timestamp}"
    train_paths, total_samples = prepare_data_partitions(
        spark,
        data_source_config["train_path"],
        training_config["num_processes"],
        train_temp_dir,
    )

    final_output_dir = (
        f"{artifact_storage_config['output_dir']}_{model_type}_{timestamp}"
    )
    distributor_args["artifact_storage_paths"]["output_dir"] = final_output_dir

    num_workers = training_config["num_processes"]
    files_per_worker = [train_paths[i::num_workers] for i in range(num_workers)]
    samples_per_worker = [total_samples // num_workers] * num_workers
    distributor_args["files_per_worker"] = files_per_worker
    distributor_args["samples_per_worker"] = samples_per_worker

    global_batch_size = training_config["batch_size"] * num_workers
    steps_per_epoch = total_samples // global_batch_size if global_batch_size > 0 else 0
    distributor_args["steps_per_epoch"] = steps_per_epoch

    if not any(files_per_worker):
        logger.info("[ERROR] Training data preparation failed.")
        spark.stop()
        return

    # --- Start distributed training ---
    distributor = TorchDistributor(
        num_processes=training_config["num_processes"],
        local_mode=(spark_config["deploy_mode"] == "local"),
        use_gpu=False,
    )
    result = distributor.run(training_function, args_dict=distributor_args)

    # --- Process results ---
    if isinstance(result, dict) and result.get("status") == "SUCCESS":
        logger.info(f"\n{model_type.upper()} training completed successfully!")
        evaluate_on_test_set(result, distributor_args)
    else:
        logger.info(
            f"\n{model_type.upper()} training failed! Error: {result.get('message', 'Unknown error')}"
        )

    # --- Cleanup temporary directories ---
    logger.info("\nCleaning up temporary directories...")
    test_temp_dir = f"{temp_dir_base}/test_data_{model_type}_{timestamp}"
    delete_hdfs_directory(train_temp_dir)
    delete_hdfs_directory(val_temp_dir)
    delete_hdfs_directory(test_temp_dir)

    # --- Upload Driver Log ---
    upload_log_file(driver_log_file, final_output_dir)

    spark.stop()


def evaluate_on_test_set(training_result, args):
    """Loads the best model and evaluates it on the test set."""
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


if __name__ == "__main__":
    main()
