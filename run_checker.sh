#!/bin/bash

# Script to check the Spark version on the cluster

# --- Python Environment ---
# Ensure this is the same path as in your main run.sh script
PYSPARK_PYTHON_PATH="/home/ubuntu/spark_env/bin/python"

# --- Main execution command ---
echo "Submitting Spark version check job..."
spark-submit \
--master yarn \
--deploy-mode cluster \
--conf spark.pyspark.python=${PYSPARK_PYTHON_PATH} \
--conf spark.pyspark.driver.python=${PYSPARK_PYTHON_PATH} \
version_checker.py

echo "Job finished. Please check the logs for the SPARK VERSION."
