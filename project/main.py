"""
Main entry point for the training pipeline.
"""

import os
from pyspark.ml.feature import StringIndexerModel
from pyspark.ml.torch.distributor import TorchDistributor
from datetime import datetime
import tempfile

from project.data.spark_utils import init_spark
from project.utils.config import load_config, parse_cli_args
from project.data.data_preprocessor import prepare_data_partitions
from project.training.trainer import training_function
from project.utils.hdfs_utils import delete_hdfs_directory, upload_log_file
from project.utils.logger import setup_logger
from project.training.evaluator import evaluate_on_test_set

# Initialize logger for the unique name for driver
driver_log_file = os.path.join(
    tempfile.gettempdir(),
    f"driver_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
)
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
    
    distributor_args["logger_file"] = driver_log_file

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

    if cli_args.output_dir:
        final_output_dir = cli_args.output_dir
        if not any(char.isdigit() for char in os.path.basename(final_output_dir)):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_output_dir = f"{final_output_dir}_{model_type}_{timestamp}"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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

    # Show prepared configuration
    logger.info("\n[Prepared Configuration for Training:")
    for key, value in distributor_args.items():
        if key not in ["files_per_worker", "samples_per_worker"]:
            logger.info(f"  {key}: {value}")

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


if __name__ == "__main__":
    main()
