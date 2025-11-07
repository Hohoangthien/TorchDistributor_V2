#!/bin/bash

echo "====== YARN NodeManagers Status ======"
yarn node -list

echo -e "\n====== YARN Applications Running ======"
yarn application -list

echo -e "\n====== YARN Applications FINISHED  ======"
yarn application -list -appStates FINISHED
echo -e "\n====== YARN Cluster Metrics ======"
curl -s "http://master:8088/ws/v1/cluster/metrics" | jq .

echo -e "\n====== Spark Application Running (by yarn-client) ======"
ps -ef | grep org.apache.spark.deploy.yarn.Client | grep -v grep

echo -e "\n====== Spark Executors Running (by CoarseGrainedExecutorBackend) ======"
ps -ef | grep CoarseGrainedExecutorBackend | grep -v grep

echo -e "\n====== Spark Configurations ======"
spark-submit --version

echo -e "\n====== Hadoop Configurations ======"
hadoop version

echo -e "\n====== HDFS Summary ======"
hdfs dfsadmin -report

echo -e "\n====== Java Version ======"
java -version

echo -e "\n====== Python Version ======"
python3 --version
