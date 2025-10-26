MODE=$1  # "client" or "cluster"
MODE_SPARK="spark-cluster3" # Using the cluster-specific spark configuration

if [ "$MODE" == "cluster" ]; then
  MODE_SPARK="spark-cluster3"
elif [ "$MODE" == "client" ]; then
  MODE_SPARK="spark-client"
else
  echo "Error: Invalid mode specified. Use 'client' or 'cluster'."
  exit 1
fi

CONFIG_FILE="config.yaml"

MODEL_TYPE=$(yq ."training.model_type" "$CONFIG_FILE")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="hdfs://master:9000/usr/ubuntu/${MODEL_TYPE}_${MODE_SPARK}_${TIMESTAMP}"

SPARK_MASTER=$(yq ."$MODE_SPARK.master" "$CONFIG_FILE")
DEPLOY_MODE=$(yq ."$MODE_SPARK.deploy_mode" "$CONFIG_FILE")
NUM_EXECUTORS=$(yq ."$MODE_SPARK.num_executors" "$CONFIG_FILE")
EXECUTOR_MEMORY=$(yq ."$MODE_SPARK.executor_memory" "$CONFIG_FILE")
EXECUTOR_CORES=$(yq ."$MODE_SPARK.executor_cores" "$CONFIG_FILE")
DRIVER_MEMORY=$(yq ."$MODE_SPARK.driver_memory" "$CONFIG_FILE")
EXECUTOR_MEMORY_OVERHEAD=$(yq ."$MODE_SPARK.executor_memory_overhead" "$CONFIG_FILE")
PYSPARK_PYTHON_PATH=$(yq ."$MODE_SPARK.python_env" "$CONFIG_FILE")

echo "Output directory set to: $OUTPUT_DIR"