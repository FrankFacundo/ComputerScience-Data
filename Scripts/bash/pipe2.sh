# set -x

function isPidStillRunning() {
    local PID=$1
    if [[ $(ps -o pid= -p $PID) -ne 0 ]];then
        echo true
    else
        echo false
    fi
}

function copyLogsWatcher() {
    local PID=$1
    echo PID parameter $PID
    local timer=$SECONDS
    local MINUTE_IN_SECONDS=60
    local COPY_FRECUENCY_MINUTES=$((30*$MINUTE_IN_SECONDS))
    while true
    do
        if [ $(isPidStillRunning "$PID") == true ];then
            echo PID_EXISTS
            DURATION=$(( $SECONDS - $timer ))
            if [ $DURATION -ge $COPY_FRECUENCY_MINUTES ];then
                timer=$SECONDS
                copyLogs
            fi
        else
            echo PID OF PUB SUB API DOES NOT EXISTS ANYMORE. MAKING LAST COPY.
            copy_logs
            break
        fi
        sleep 2
    done
}

function copyLogs() {
    echo COPYING...
}
