set -x

# This will create the master server on http://127.0.0.1:8080/
# if the port does not work, check the master process with `ps ax | grep spark`
start-master.sh
# This will create the local worker.
start-worker.sh spark://$HOSTNAME:7077
