# version_checker.py
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession.builder.appName("SparkVersionChecker").getOrCreate()
    print("="*60)
    print(f"SPARK VERSION ON CLUSTER: {spark.version}")
    print("="*60)
    spark.stop()