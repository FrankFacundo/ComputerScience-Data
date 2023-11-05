# Spark

For installation:
https://phoenixnap.com/kb/install-spark-on-ubuntu

When finish check versions:

- Java: java --version
- Scala: scala
- Spark: spark-shell
- Pyspark: pip show pyspark

and check env variables:

- SPARK_HOME
- PATH
- PYSPARK_PYTHON

if empty, fill them correctly according to your PC.

- export `SPARK_HOME=/opt/spark`
- export `PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin`
- export `PYSPARK_PYTHON=/usr/bin/python3`

To start Spark execute:

    start-master.sh
    This will create a server on http://127.0.0.1:8080/
    start-worker.sh spark://$(hostname):7077

To stop master:

    stop-master.sh

To stop worker:

    1) Find the process id:
        `ps ax | grep spark`
    2) Kill it:
        `sudo kill pid1`
