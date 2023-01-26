SCRIPT_PATH=$(realpath $0)
SCRIPT_BASE_PATH=$(dirname "${SCRIPT_PATH}")
echo Base path : $SCRIPT_BASE_PATH

source ${SCRIPT_BASE_PATH}/pipe2.sh

echo pid current1 $$
echo pid current1 $BASHPID

copyLogsWatcher $$ &
sleep 30
echo END
