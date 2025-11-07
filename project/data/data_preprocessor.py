import os
from urllib.parse import urlparse
import pyarrow.fs
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.ml.linalg import Vector

# Define a UDF to consistently convert Vector to a Python list.
def vector_to_list(v: Vector) -> list:
    if v is None:
        return []
    return v.toArray().tolist()

vector_to_array_udf = udf(vector_to_list, ArrayType(DoubleType()))

def prepare_data_partitions(spark, data_path, num_processes, output_temp_dir):

    print(
        f"[DRIVER] Preparing data from {data_path} into {num_processes} partitions -> {output_temp_dir}"
    )
    try:
        df = spark.read.parquet(data_path)
        
        total_count = df.count()

        # Standardize the feature column from Vector to ArrayType before writing
        df_standardized = df.withColumn(
            "features_array", vector_to_array_udf(col("scaled_features"))
        ).drop("scaled_features").withColumnRenamed("features_array", "scaled_features")
        
        df_standardized.repartition(num_processes).write.mode('overwrite').parquet(output_temp_dir)
        
        print(f"Successfully wrote standardized data to {output_temp_dir}")
        
        # Discover the paths of the written part-files using Spark's underlying Hadoop FS API
        # This is more robust than using pyarrow.fs directly, as it uses Spark's configured filesystem.
        sc = spark.sparkContext
        path = sc._jvm.org.apache.hadoop.fs.Path(output_temp_dir)
        fs = path.getFileSystem(sc._jsc.hadoopConfiguration())
        
        file_statuses = fs.listStatus(path)
        parquet_files = [
            str(f.getPath()) for f in file_statuses if f.getPath().getName().endswith('.parquet')
        ]
        
        print(f"Discovered {len(parquet_files)} Parquet part-files.")
        
        return parquet_files, total_count
        
    except Exception as e:
        import traceback
        print(f"[DRIVER] Data preparation failed for {data_path}: {e}")
        traceback.print_exc()
        return [], 0
