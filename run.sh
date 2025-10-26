#!/bin/bash


# --- Configuration ---
MODE=$1  # "client" or "cluster"

CONFIG_FILE="config.yaml"
MODE_SPARK="spark-cluster3" 
ALLUXIO_CLIENT_JAR="/usr/local/alluxio/client/alluxio-2.9.4-client.jar"

if [ "$MODE" == "cluster" ]; then
  MODE_SPARK="spark-cluster3"
elif [ "$MODE" == "client" ]; then
  MODE_SPARK="spark-client"
else
  echo "Error: Invalid mode specified. Use 'client' or 'cluster'."
  exit 1
fi

# --- Dynamic Output Directory ---
MODEL_TYPE=$(yq ."training.model_type" "$CONFIG_FILE")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="hdfs://master:9000/usr/ubuntu/${MODEL_TYPE}_${MODE_SPARK}_${TIMESTAMP}"

echo "===================================================================="
echo "Starting Spark Cluster Job"
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
echo "Packaging project files into project-cluster.zip..."
zip -r project-cluster.zip project/ -x "*__pycache__*" "*.pyc"

echo "Submitting Spark job in cluster mode..."
spark-submit \
--master ${SPARK_MASTER} \
--deploy-mode ${DEPLOY_MODE} \
--num-executors ${NUM_EXECUTORS} \
--executor-memory ${EXECUTOR_MEMORY} \
--executor-cores ${EXECUTOR_CORES} \
--driver-memory ${DRIVER_MEMORY} \
--py-files project-cluster.zip \
--files ${CONFIG_FILE} \
--conf spark.executor.memoryOverhead=${EXECUTOR_MEMORY_OVERHEAD} \
${SPARK_JARS_CONF} \
--conf spark.pyspark.python=${PYSPARK_PYTHON_PATH} \
--conf spark.pyspark.driver.python=${PYSPARK_PYTHON_PATH} \
project/main.py --config ${CONFIG_FILE} --output_dir "${OUTPUT_DIR}"

EXIT_CODE=$?

# --- Post-processing and Cleanup ---
echo "Spark job finished with exit code: $EXIT_CODE"
rm project-cluster.zip

if [ $EXIT_CODE -eq 0 ]; then
    echo "Training job successful."    
    echo "Listing final contents of the output directory in HDFS:"
    hdfs dfs -ls -R "${OUTPUT_DIR}"
    hdfs dfs -get "${OUTPUT_DIR}" saves/cluster/

else
    echo "Training job failed. Check the logs for details."
fi

exit $EXIT_CODE