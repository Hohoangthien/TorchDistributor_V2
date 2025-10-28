import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from datetime import timedelta
import time
import itertools
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau

from project.utils.logger import setup_logger
from project.models import create_model
from project.data.data_loader import create_pytorch_dataloader
from project.utils.visualization import plot_and_save_history
from project.utils.hdfs_utils import upload_log_file, upload_local_directory_to_hdfs
from project.training.evaluator import evaluate_loop


def training_function(args_dict):
    rank = int(os.environ.get("RANK", 0))
    model_type = args_dict["model_type"]

    # --- Conditional Logger Initialization ---
    log_file = None
    if rank == 0:
        log_file = f"/tmp/training_{model_type}_rank_0.log"
    logger = setup_logger(rank, log_file=log_file)

    try:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if world_size > 1 and not dist.is_initialized():
            dist.init_process_group(
                backend="gloo", init_method="env://", timeout=timedelta(minutes=30)
            )

        # --- Unpack arguments for clarity ---
        model_type = args_dict["model_type"]
        batch_size = args_dict["batch_size"]
        max_epochs = args_dict["epochs"]
        steps_per_epoch = args_dict.get("steps_per_epoch", 0)

        # Separate data source (read) and artifact storage (write) paths
        data_source_paths = args_dict["data_source_paths"]
        artifact_storage_paths = args_dict["artifact_storage_paths"]
        final_output_dir = artifact_storage_paths["output_dir"]  # This is the HDFS path

        # --- Data Loader Initialization ---
        logger.info("Initializing Dataloaders...")
        worker_files = args_dict["files_per_worker"][rank]
        num_samples_for_worker = args_dict["samples_per_worker"][rank]

        train_dataloader = create_pytorch_dataloader(
            worker_files, batch_size, num_samples_for_worker, is_training=True
        )
        if not train_dataloader:
            return {"status": "NO_DATA", "rank": rank}

        val_dataloader = None
        if rank == 0 and args_dict.get("validation_file_path"):
            val_dataloader = create_pytorch_dataloader(
                [args_dict["validation_file_path"]],
                batch_size * 2,
                num_samples=0,
                is_training=False,
            )
        logger.info("Dataloaders initialized.")

        # --- Model and Optimizer ---
        device = torch.device("cpu")
        model = create_model(**args_dict).to(device)
        if world_size > 1:
            model = nn.parallel.DistributedDataParallel(model)
        optimizer = optim.AdamW(
            model.parameters(), lr=args_dict.get("learning_rate", 0.002)
        )
        
        scheduler = None
        scheduler_config = args_dict.get("training", {}).get("scheduler", {})
        if scheduler_config.get("enabled", False) and rank == 0:
            logger.info("Learning rate scheduler (ReduceLROnPlateau) is enabled.")
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=scheduler_config.get("factor", 0.1),
                patience=scheduler_config.get("patience", 3),
                min_lr=scheduler_config.get("min_lr", 1e-6),
                verbose=True
            )

        criterion = nn.CrossEntropyLoss(reduction="none")
        eval_criterion = nn.CrossEntropyLoss()

        # --- Training Loop ---
        logger.info(f"Starting training for {model_type.upper()} model...")
        best_val_acc = 0.0
        history = {
            "train_losses": [],
            "train_accuracies": [],
            "val_losses": [],
            "val_accuracies": [],
        }

        for epoch in range(max_epochs):
            logger.info(f"Starting Epoch {epoch + 1}/{max_epochs}")
            model.train()
            for i, (features, labels, weights) in enumerate(
                itertools.islice(train_dataloader, steps_per_epoch)
            ):
                features, labels, weights = (
                    features.to(device),
                    labels.to(device),
                    weights.to(device),
                )
                optimizer.zero_grad(set_to_none=True)
                outputs = model(features)
                loss = criterion(outputs, labels)
                weighted_loss = (loss * weights).mean()
                weighted_loss.backward()
                optimizer.step()

            if world_size > 1:
                dist.barrier()

            # --- Validation and Checkpointing (on rank 0) ---
            if rank == 0:
                logger.info("Starting validation...")
                model_to_eval = model.module if world_size > 1 else model
                val_loss, val_acc, _, _ = evaluate_loop(
                    model_to_eval, val_dataloader, eval_criterion, device
                )
                logger.info(
                    f"EPOCH {epoch + 1} SUMMARY: Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
                )

                history["val_losses"].append(val_loss)
                history["val_accuracies"].append(val_acc)
                history["train_losses"].append(0)  # Placeholder
                history["train_accuracies"].append(0)  # Placeholder

                if scheduler:
                    scheduler.step(val_loss)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    logger.info(
                        f"New best val_acc: {best_val_acc:.4f}. Saving model..."
                    )
                    torch.save(
                        model_to_eval.state_dict(), f"/tmp/best_{model_type}_model.pth"
                    )

                # Use the correct HDFS output path for saving artifacts
                plot_and_save_history(
                    **history, output_dir=final_output_dir, model_type=model_type
                )

        # --- Final Upload (on rank 0) ---
        if rank == 0:
            local_model_path = f"/tmp/best_{model_type}_model.pth"
            if os.path.exists(local_model_path):
                # Create a temporary directory for the model to be uploaded
                model_upload_dir = "/tmp/model_upload"
                os.makedirs(model_upload_dir, exist_ok=True)
                os.rename(local_model_path, os.path.join(model_upload_dir, os.path.basename(local_model_path)))
                logger.info(f"Uploading model from {model_upload_dir} to {final_output_dir}")
                upload_local_directory_to_hdfs(model_upload_dir, final_output_dir)

            upload_log_file(log_file, final_output_dir)

        if world_size > 1:
            dist.barrier()

        return {"status": "SUCCESS", "best_val_accuracy": best_val_acc}

    except Exception as e:
        import traceback

        logger.error(f"An unhandled exception occurred on RANK {rank}: {e}")
        logger.error(traceback.format_exc())
        return {"status": "ERROR", "message": str(e)}
