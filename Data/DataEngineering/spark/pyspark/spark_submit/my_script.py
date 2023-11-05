# Import SparkContext and SparkConf
from pyspark import SparkContext, SparkConf

# Create Spark configuration object
conf = SparkConf().setAppName("AverageCalculator")

# Create SparkContext object
sc = SparkContext(conf=conf)

# Define a function to parse the data
def parse_data(line):
    try:
        return [float(x) for x in line.split(',')]
    except Exception as e:
        return []

# Initialize a list of numbers
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Parallelize the list to create an RDD
numbers_rdd = sc.parallelize(numbers)

# Calculate the total and count
total_and_count = numbers_rdd.map(lambda x: (x, 1)).reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]))

# Calculate the average
average = total_and_count[0] / total_and_count[1]

# Print the result
print("The average is:", average)

# Stop the SparkContext
sc.stop()
