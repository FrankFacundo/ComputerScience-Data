from pyspark.sql import SparkSession

# Initialize a Spark session
spark = SparkSession.builder.appName("SimpleApp").getOrCreate()

# Example data processing
data = [("John", 28), ("Smith", 44), ("Adam", 33)]
df = spark.createDataFrame(data, ["Name", "Age"])
df.show()

# Stop the Spark session
spark.stop()
