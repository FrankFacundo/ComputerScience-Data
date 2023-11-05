For installation:
https://phoenixnap.com/kb/install-spark-on-ubuntu

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
