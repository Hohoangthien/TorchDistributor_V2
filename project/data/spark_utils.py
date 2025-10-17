from pyspark.sql import SparkSession

def init_spark(mode):
    """Khởi tạo Spark Session"""
    if mode == "local":
        return (
            SparkSession.builder.appName("Multi_Model_NIDS_Local_Training")
            .master("local[*]")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .getOrCreate()
        )
    else:
        return (
            SparkSession.builder.appName("Multi_Model_NIDS_Cluster_Training")
            .master("yarn")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .getOrCreate()
        )
