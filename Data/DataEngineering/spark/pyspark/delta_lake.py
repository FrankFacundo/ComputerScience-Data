from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("DeltaLakeWithPySpark") \
    .config("spark.jars.packages", "io.delta:delta-core_2.12:1.0.0") \  # Adjust the version as needed
    .getOrCreate()
