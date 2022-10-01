# Pyspark 2.3.0

This cheatsheet is code oriented so to see spark concepts please check spark_notions.md

## Session

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder\
    .config()\
    .enableHiveSupport().getOrCreate()
```

## Base types

```python
from pyspark.sql.types import LongType, DataType, StringType, IntegerType
```

## Container types

```python
from pyspark.sql.types import ArrayType, StructType

```

## Files

### Read

Import a hard drive file or a HDFS file.

- Method 1 - Quite usefull in spark shell to make little and fast requests.

* `df` is a DataFrame spark object.

```python
from pyspark.sql import SQLContext

df = sqlContext.sql("SELECT * FROM orc.`path_orc_or_directory` WHERE col1 = 'value1' limit 500")
```

- Method 2

```python
df = spark.read.csv('csv_path', sep=';', header=True)
```

### Write

* coalesce define the number of partitions for a dataframe. In this example df is stocked in a single file.

```python
df.coalesce(1).write.format('orc').mode('overwrite').save('directory_path')
```

## Create a Dataframe

To create an array:

```python
arrayData = [
        ('James',['Java','Scala'], [20,21]),
        ('Michael',['Spark','Java',"Python"], [15,16,17]),
        ('Robert',['CSharp','Go'], [10,11]),
        ('Juan',["CPP"],[1]),
        ('Jefferson',['R','Scala'],[2, 4])]

df1 = spark.createDataFrame(data=arrayData, schema = ['name','knownLanguages','properties'])

```

## DAG

To check Schema.

```python
data.printSchema()
```

## Selecting Data

* In Python way, always put parentheses for each boolean operation.

### Boolean operators - First Python way - Second SQL way

* Negation: `~` , `NOT`
* And: `&` , `AND`
* Or: `|` , `OR`
* Comparison: `==`

### Like

* `%`: equivalent to `*` in regex

### Select

```python
df.select('*').where(df['col1'] == 'TEST').show()
df.select('*').where(df['col1'].isNotNull()).show()
df.select(df.col1, df.col2).show()
df.select(df['col1'], df['col2']).show()
```

### Filtering

Method 1 - Python way

```python
import pyspark.sql.functions as f

df.filter(
    (~(f.col("col1").like("init_%")) & (f.col("col2") == "value"))
)
```

Method 2 - SQL way

```python
df.filter(
    "((NOT col1 LIKE init_% AND (col2 = value)))"
)
```

### Useful

Show distinct values.

```python
df.distinct()
```

Show 20 rows and it does not truncate results.

```python
n = 20
df.show(n, truncate=False)
```

## Cast columns in Dataframes

Example:

```python
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.sql.functions import split

df = df.withColumn("colListInStringSepByComma", split(df.colListInStringSepByComma, ","))
df = df.withColumn("colListInStringSepByComma", df.colListInStringSepByComma.cast(ArrayType(DoubleType()))) 
df = df.withColumnRenamed("colListInStringSepByComma", "colArray")
```
