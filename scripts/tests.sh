#!/bin/bash

# This script in used on the continuous integration platform to execute the tests
# and extract some information about the results
#
# Usage: ./tests.sh --run |
#                   --count-tests | --count-failures | --count-disabled
#                   --count-errors | --count-binaries | --count-reports
#
# With --run, it runs each file which matches "bin/*_test{,d}", and outputs
# the results in a JUnit XML file stored in $PWD/test-reports/
#
# E.g. bin/foo_test will produce test-reports/foo_test.xml


if [ $# -ne 1 ]; then
    echo "$0: one argument expected, $# provided" >&2
    exit 1
fi

get-tests()
{
    test_files=$(ls bin/*_test 2> /dev/null)
    testd_files=$(ls bin/*_testd 2> /dev/null)
    test_exe_files=$(ls bin/*_test.exe 2> /dev/null)
    testd_exe_files=$(ls bin/*_testd.exe 2> /dev/null)
    echo $test_files $testd_files $test_exe_files $testd_exe_files
}

run-tests ()
{
    # Create or empty out the output directory
    if [ -d test-reports ]; then
        echo "$0: deleting content of test-reports/"
        rm -f test-reports/*
    else
        mkdir test-reports
    fi
    # Check the existence of test programs
    if [ -z "$(get-tests)" ]; then
        echo "$0: no test executable found"
        exit 0
    fi
    # Run each test
    for test in $(get-tests); do
        output_file=test-reports/`basename "$test" .exe`.xml
        "$test" --gtest_output=xml:"$output_file"
        exit_code="$?"
        # Check the test executable didn't crash
        if [ -e "$output_file" ]; then
            # Little fix: Googletest marks skipped tests with a 'status="notrun"' attribute,
            # but the JUnit XML understood by Jenkins requires a '<skipped/>' element instead.
            # source: http://stackoverflow.com/a/14074664
            sed -i 's:\(<testcase [^>]*status="notrun".*\)/>:\1><skipped/></testcase>:' "$output_file"
        else
            echo "$0: $output_file was not created; $test ended with code $exit_code" >&2
        fi
    done
}


# Fetch the <testsuites> XML elements in test-reports/*.xml,
# extract and sum the attribute given in argument
# This function relies on the element being written on a single line:
# E.g. <testsuites tests="212" failures="4" disabled="0" errors="0" ...
sum-attribute-from-testsuites ()
{
    # Check the existence of report files
    if ! ls test-reports/*.xml &> /dev/null; then
        echo "$0: no test report found" >&2
        echo 0
        return
    fi
    attribute="$1"
    # grep the lines containing '<testsuites'; for each one, match the 'attribute="..."' pattern, and collect the "..." part
    counts=`sed -ne "s/.*<testsuites[^>]* $attribute=\"//" \
                 -e "/^[0-9]/s/\".*//p" test-reports/*.xml`
    # sum the values
    total=0
    for value in $counts; do
        total=$(( $total + $value ))
    done
    echo "$total"
}

case "$1" in
    --run )
        run-tests
        ;;
    --count-tests )
        sum-attribute-from-testsuites tests
        ;;
    --count-failures )
        sum-attribute-from-testsuites failures
        ;;
    --count-disabled )
        sum-attribute-from-testsuites disabled
        ;;
    --count-errors )
        sum-attribute-from-testsuites errors
        ;;
    --count-binaries )
        get-tests | wc -w | tr -d ' '
        ;;
    --count-reports )
        ls test-reports/*.xml 2> /dev/null | wc -l | tr -d ' '
        ;;
    * )
        echo "$0: unexpected argument: $1"
        ;;
esac

