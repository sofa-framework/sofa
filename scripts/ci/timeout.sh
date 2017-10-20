#!/bin/bash

# Here we define a 'run-command-with-timeout' function, because Windows doesn't
# have a 'timeout' command.

# Usage: timeout.sh <id> <command> <number-of-seconds>
#
# This runs <command> with a timeout of <number-of-seconds>.  If
# <command> terminates before the timeout, save the exit code in
# <id>.exit_code, otherwise create <id>.timeout

usage() {
    # Runs <command> with a timeout of <number-of-seconds>.  If
    # <command> terminates before the timeout, save the exit code in
    # <id>.exit_code, otherwise create <id>.timeout
    echo "Usage: timeout.sh <id> <command> <number-of-seconds>"
}

if [[ "$#" != 3 ]]; then
    usage
    exit 1
fi

# usage: run-command <id> <cmd>
# Runs <cmd>, saves the pid of the spawned shell in <id>.pid and saves the exit
# code in <id>.exit_code.
run-command() {
    bash -c "echo "\$\$" > $1.pid; $2" && local exit_code=$? || local exit_code=$?
    echo $exit_code > $1.exit_code
}
# usage: executioner <id> <n>
# Sleeps <n> seconds, creates <id>.timeout, and kills <id>.pid.
executioner() {
    sleep $2
    touch $1.timeout
    kill $(cat $1.pid) >& /dev/null || true
}
# usage: run-command-with-timeout <id> <cmd> <n>
# Runs <cmd> with a timeout of <n> seconds.  If <cmd> terminates before the
# timeout, save the exit code in <id>.exit_code, otherwise create <id>.timeout
run-command-with-timeout() {
    rm -f $1.exit_code
    rm -f $1.timeout
    run-command $1 "$2" &
    local cmd_pid=$!
    executioner $1 $3 & local executioner_pid=$!
    wait $cmd_pid >& /dev/null || true
    kill $executioner_pid || true
    rm -f $1.pid
}

timeout "$3" bash -c "$2"
exit_code=$?
if [[ ($(uname) = Darwin && $exit_code = 137 ) || ( $exit_code = 124 ) ]]; then
    touch "$1".timeout
fi
echo $exit_code > "$1".exit_code
