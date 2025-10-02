print("Initializing Spark session...")

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("TestSpark") \
    .config("spark.submit.deployMode", "client") \
    .config("spark.excludeOnFailure.enabled", "true") \
    .master("yarn") \
    .getOrCreate()
    
train_path1 = "alluxio:/usr/ubuntu/data/classweights-43-21/train_df.parquet"

train_df = spark.read.parquet(train_path1).cache()
df_large = train_df
for _ in range(999):  # Lặp lại 1000 lần
    df_large = df_large.union(train_df)


print(df_large.count())


df_large.write.mode('overwrite').parquet("alluxio:/usr/ubuntu/data/classweights-43-21/big_train_df.parquet")