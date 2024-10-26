#!/bin/bash

date=`date '+%Y-%m-%dT%H:%M:%S'`
testName="headlessRecorder_test"
arg0=$0
arg1=$1
output="${arg1#*:}"

tests=0
failures=0
errors=0
timer=0
testOutput=""
failureString=""
echo "$arg0"
echo "OK"

# first test, check result file size
run_test()
{
    local fileSize=0
    local timeSecond=0
    local timeMilliSecond=0
    local minimalSize=30000 # should be greater than 27kb
    local currentTestName="size_check"

    echo "[ RUN      ] $testName.$currentTestName"
    STARTTIME=$(date +%s%N | cut -b1-13)
    ${arg0/$testName/runSofa} -g hRecorder --video --fps=60 -a --filename tmp
    ENDTIME=$(date +%s%N | cut -b1-13)
    if [ -e tmp.avi ]; then
        fileSize=$(stat -c%s "tmp.avi")
        rm tmp.avi
    fi
    tests=$((tests + 1))

    if [ $fileSize -ge $minimalSize ]; then
        echo "[       OK ] $testName.$currentTestName ( ms)"
    else
        failureString="[ERROR] $testName.$currentTestName ($(($ENDTIME - $STARTTIME)) ms) size too small: $fileSize"
        echo $failureString
        failures=$((failures + 1))
    fi
    timer=$((timer + $ENDTIME - $STARTTIME))
}

run_test > run_test_output.tmp 2>&1
testOutput=$(cat run_test_output.tmp)
rm -f run_test_output.tmp
echo "$testOutput"

time=$(($timer/1000))"."$(($timer%1000))
echo "<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<testsuites tests=\"1\" failures=\"$failures\" disabled=\"0\" errors=\"$errors\" timestamp=\"$date\" time=\"$time\" name=\"AllTests\">
<testsuite name=\"$testName\" tests=\"$tests\" failures=\"$failures\" disabled=\"0\" errors=\"$errors\" time=\"$time\">
<testcase name=\"script\" status=\"run\" time=\"$time\" classname=\"$testName\">" > $output

if [ $failures -gt 0 ]; then
    echo "<failure message=\"$failureString\" type=\"\"><![CDATA[$testOutput]]></failure>" >> $output
fi

echo "</testcase>
</testsuite>
</testsuites>" >> $output
