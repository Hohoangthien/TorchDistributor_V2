#!/bin/bash

# Script to run the Spark training job using a configuration file.

# --- CRITICAL: ALLUXIO INTEGRATION ---
ALLUXIO_CLIENT_JAR="/usr/local/alluxio/client/alluxio-2.9.4-client.jar"
# --- Application Configuration ---
MODE_SPARK="spark-cluster3"
# MODE_SPARK="spark-client"

CONFIG_FILE="config.yaml"

# --- Load Spark parameters from config.yaml ---
if command -v yq >/dev/null 2>&1; then
    SPARK_MASTER=$(yq ."$MODE_SPARK.master" "$CONFIG_FILE")
    DEPLOY_MODE=$(yq ."$MODE_SPARK.deploy_mode" "$CONFIG_FILE")
    NUM_EXECUTORS=$(yq ."$MODE_SPARK.num_executors" "$CONFIG_FILE")
    EXECUTOR_MEMORY=$(yq ."$MODE_SPARK.executor_memory" "$CONFIG_FILE")
    EXECUTOR_CORES=$(yq ."$MODE_SPARK.executor_cores" "$CONFIG_FILE")
    DRIVER_MEMORY=$(yq ."$MODE_SPARK.driver_memory" "$CONFIG_FILE")
    EXECUTOR_MEMORY_OVERHEAD=$(yq ."$MODE_SPARK.executor_memory_overhead" "$CONFIG_FILE")
    PYSPARK_PYTHON_PATH=$(yq ."$MODE_SPARK.python_env" "$CONFIG_FILE")
else
    echo "yq is required to parse config.yaml. Please install yq."
    exit 1
fi

# --- Dynamic JAR loading logic ---
SPARK_JARS_CONF=""
if grep -q "alluxio://" "$CONFIG_FILE"; then
  echo "Alluxio path detected in config. Preparing Alluxio JARs..."
  if [ ! -f "$ALLUXIO_CLIENT_JAR" ]; then
    echo "Error: Alluxio path detected, but JAR file not found at $ALLUXIO_CLIENT_JAR"
    echo "Please update the ALLUXIO_CLIENT_JAR variable in this script."
    exit 1
  fi
  # Construct the spark-submit options for Alluxio
  SPARK_JARS_CONF="--jars ${ALLUXIO_CLIENT_JAR} --conf spark.driver.extraClassPath=${ALLUXIO_CLIENT_JAR} --conf spark.executor.extraClassPath=${ALLUXIO_CLIENT_JAR}"
  echo "Alluxio JARs configured."
else
  echo "No Alluxio path detected. Running with default HDFS client."
fi

# --- Packaging the project ---
echo "Packaging project files into project.zip..."
zip -r project.zip project/ -x "*__pycache__*" "*.pyc"

# --- Main execution command ---
echo "Submitting Spark job..."
spark-submit \
--master ${SPARK_MASTER} \
--deploy-mode ${DEPLOY_MODE} \
--num-executors ${NUM_EXECUTORS} \
--executor-memory ${EXECUTOR_MEMORY} \
--executor-cores ${EXECUTOR_CORES} \
--driver-memory ${DRIVER_MEMORY} \
--py-files project.zip \
--files ${CONFIG_FILE} \
--conf spark.executor.memoryOverhead=${EXECUTOR_MEMORY_OVERHEAD} \
${SPARK_JARS_CONF} \
--conf spark.pyspark.python=${PYSPARK_PYTHON_PATH} \
--conf spark.pyspark.driver.python=${PYSPARK_PYTHON_PATH} \
project/main.py --config ${CONFIG_FILE}

# --- Cleanup ---
EXIT_CODE=$?
echo "Cleaning up packaged file..."
rm project.zip

echo "Spark job finished with exit code: $EXIT_CODE"
exit $EXIT_CODE
