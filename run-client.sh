#!/bin/bash

# This script is designed to run the training pipeline in client mode.
# It constructs the output directory path dynamically and passes it to the main script.

# --- Configuration ---
CONFIG_FILE="config.yaml"
MODE_SPARK="spark-client" # Using the client-specific spark configuration
ALLUXIO_CLIENT_JAR="/usr/local/alluxio/client/alluxio-2.9.4-client.jar"

# Check if yq is installed
if ! command -v yq >/dev/null 2>&1; then
    echo "Error: yq is required to parse config.yaml, but it's not installed." >&2
    echo "Please install yq (e.g., 'pip install yq' or 'sudo snap install yq')." >&2
    exit 1
fi

# --- Dynamic Output Directory ---
# Generate the HDFS output directory path based on model type and current timestamp.
MODEL_TYPE=$(yq ."training.model_type" "$CONFIG_FILE")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="hdfs://master:9000/usr/ubuntu/${MODEL_TYPE}_client_${TIMESTAMP}"

echo "===================================================================="
echo "Starting Spark Client Job"
echo "Config File:      $CONFIG_FILE"
echo "Model Type:       $MODEL_TYPE"
echo "Generated HDFS Output Dir: $OUTPUT_DIR"
echo "===================================================================="

# --- Load Spark parameters from config.yaml ---
SPARK_MASTER=$(yq ."$MODE_SPARK.master" "$CONFIG_FILE")
DEPLOY_MODE=$(yq ."$MODE_SPARK.deploy_mode" "$CONFIG_FILE")
NUM_EXECUTORS=$(yq ."$MODE_SPARK.num_executors" "$CONFIG_FILE")
EXECUTOR_MEMORY=$(yq ."$MODE_SPARK.executor_memory" "$CONFIG_FILE")
EXECUTOR_CORES=$(yq ."$MODE_SPARK.executor_cores" "$CONFIG_FILE")
DRIVER_MEMORY=$(yq ."$MODE_SPARK.driver_memory" "$CONFIG_FILE")
EXECUTOR_MEMORY_OVERHEAD=$(yq ."$MODE_SPARK.executor_memory_overhead" "$CONFIG_FILE")
PYSPARK_PYTHON_PATH=$(yq ."$MODE_SPARK.python_env" "$CONFIG_FILE")

# --- Dynamic JAR loading logic ---
SPARK_JARS_CONF=""
if grep -q "alluxio://" "$CONFIG_FILE"; then
  echo "Alluxio path detected in config. Preparing Alluxio JARs..."
  if [ ! -f "$ALLUXIO_CLIENT_JAR" ]; then
    echo "Error: Alluxio path detected, but JAR file not found at $ALLUXIO_CLIENT_JAR"
    echo "Please update the ALLUXIO_CLIENT_JAR variable in this script."
    exit 1
  fi
  SPARK_JARS_CONF="--jars ${ALLUXIO_CLIENT_JAR} --conf spark.driver.extraClassPath=${ALLUXIO_CLIENT_JAR} --conf spark.executor.extraClassPath=${ALLUXIO_CLIENT_JAR}"
  echo "Alluxio JARs configured."
else
  echo "No Alluxio path detected. Running with default HDFS client."
fi

# --- Packaging the project ---
echo "Packaging project files into project-client.zip..."
zip -r project-client.zip project/ -x "*__pycache__*" "*.pyc"

# --- Main execution command ---
# The --output_dir argument overrides the setting in the config file.

echo "Submitting Spark job in client mode..."
spark-submit \
--master ${SPARK_MASTER} \
--deploy-mode ${DEPLOY_MODE} \
--num-executors ${NUM_EXECUTORS} \
--executor-memory ${EXECUTOR_MEMORY} \
--executor-cores ${EXECUTOR_CORES} \
--driver-memory ${DRIVER_MEMORY} \
--py-files project-client.zip \
--files ${CONFIG_FILE} \
--conf spark.executor.memoryOverhead=${EXECUTOR_MEMORY_OVERHEAD} \
${SPARK_JARS_CONF} \
--conf spark.pyspark.python=${PYSPARK_PYTHON_PATH} \
--conf spark.pyspark.driver.python=${PYSPARK_PYTHON_PATH} \
project/main.py --config ${CONFIG_FILE} --output_dir "${OUTPUT_DIR}"

EXIT_CODE=$?

# --- Post-processing and Cleanup ---
echo "Spark job finished with exit code: $EXIT_CODE"
rm project-client.zip

if [ $EXIT_CODE -eq 0 ]; then
    echo "Training job successful."    
    echo "Listing final contents of the output directory in HDFS:"
    hdfs dfs -ls -R "${OUTPUT_DIR}"
else
    echo "Training job failed. Check the logs for details."
fi

exit $EXIT_CODE