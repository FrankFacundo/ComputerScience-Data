Note that not all options may be applicable to every Spark application, and some are specific to certain versions of Spark. Please ensure you check with the version of Spark you are using.

Application Options:

- --class: The entry point for your application (e.g., org.apache.spark.examples.SparkPi).
- --master: The master URL for the cluster (e.g., spark://23.195.26.187:7077).
- --deploy-mode: Whether to deploy your driver on the worker nodes (cluster) or locally as an external client (client).
- --conf: Arbitrary Spark configuration property (e.g., spark.executor.memory=4g).
- --properties-file: Path to a file from which to load extra properties.
- --name: A name for your application.

Resource Options:

- --driver-memory: Memory for driver (e.g., 1000M, 2G).
- --driver-java-options: Extra Java options for the driver.
- --driver-library-path: Extra library path entries for the driver.
- --driver-class-path: Extra class path entries for the driver.
- --executor-memory: Memory per executor (e.g., 2G).
- --executor-cores: The number of cores for each executor.
- --total-executor-cores: Total cores for all executors.

Environment Options:

- --driver-cores: Number of cores used by the driver, only in cluster mode.
- --supervise: If given, restarts the driver on failure.
- --queue: The YARN queue to submit to (YARN-only).
- --num-executors: Number of executors to launch (YARN-only).
- --archives: Comma-separated list of archives to be extracted into the working directory of each executor.

Spark Standalone or Mesos with cluster deploy mode only:

- --kill: To kill the given application.
- --status: To request the status of an application.

Classic Examples of Spark-Submit

Example 1: Running SparkPi in a standalone cluster

```sh
spark-submit \
  --master spark://$(hostname):7077 \
  --executor-memory 25G \
  --total-executor-cores 8 \
  my_script.py
```

```sh
spark-submit \
  --master spark://$(hostname):7077 \
  my_script.py
```

```sh
spark-submit \
  --class org.apache.spark.examples.SparkPi \
  --master spark://23.195.26.187:7077 \
  --executor-memory 2G \
  --total-executor-cores 100 \
  /path/to/examples.jar \
  1000
```

Example 2: Submitting a YARN job with extra Java options

```sh
spark-submit \
  --class com.example.MyApp \
  --master yarn \
  --deploy-mode cluster \
  --driver-memory 4g \
  --executor-memory 2g \
  --executor-cores 1 \
  --queue thequeue \
  --driver-java-options "-XX:+PrintGCDetails -XX:+PrintGCTimeStamps" \
  --conf spark.eventLog.enabled=true \
  --conf spark.eventLog.dir=hdfs://namenode:8021/directory \
  --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
  /my/app.jar \
  --app-argument 1000
```

Example 3: Using Spark-Submit with Python

```sh
spark-submit \
  --master yarn \
  --py-files pyfile.zip,myfile.py \
  --executor-memory 2G \
  --total-executor-cores 4 \
  my_script.py \
  --arg1 val1
```

When you use spark-submit, make sure that you adjust the memory, the number of executors, cores, and other configurations based on the actual requirements and limitations of your cluster or cloud environment. These settings can greatly influence the performance and resource utilization of your Spark jobs.
