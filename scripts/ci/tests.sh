#! /bin/bash

# This script runs all the googletest-based automatic tests it can find, assumed
# to be the executables in bin/ that match *_test, and saves the results in XML
# files, that can be understood by Jenkins.

set -o errexit

# Disable colored output to avoid dirtying the log
export GTEST_COLOR=no
export SOFA_COLOR_TERMINAL=no

usage() {
    echo "Usage: tests.sh (run|print-summary) <build-dir> <src-dir>"
}

if [[ "$#" = 3 ]]; then
    command="$1"
    build_dir="$2"
    src_dir="$3"
    output_dir="$build_dir/tests"
else
    usage; exit 1
fi

if [[ ! -d "$build_dir/lib/" ]]; then
    echo "Error: '$build_dir' does not look like a Sofa build."
    usage; exit 1
elif [[ ! -d "$src_dir/applications/plugins" ]]; then
    echo "Error: '$src_dir' does not look like a Sofa source tree."
    usage; exit 1
fi


export SOFA_DATA_PATH="$src_dir:$src_dir/examples:$src_dir/share"

list-tests() {
    pushd "$build_dir/bin" > /dev/null
    for file in *; do
        case "$file" in
            *_test|*_testd|*_test.exe|*_testd.exe)
                echo $file
                ;;
        esac
    done
    popd > /dev/null
}

initialize-testing() {
    echo "Initializing testing."
    rm -rf "$output_dir"
    rm -rf "$output_dir/reports"
    mkdir -p "$output_dir"
    mkdir -p "$output_dir/reports"
    list-tests | while read test; do
        echo "$test"
        mkdir -p "$output_dir/$test"
    done > "$output_dir/tests.txt"
}

fix-test-report() {
    # Little fix: Googletest marks skipped tests with a 'status="notrun"' attribute,
    # but the JUnit XML understood by Jenkins requires a '<skipped/>' element instead.
    # source: http://stackoverflow.com/a/14074664
    sed -i'.bak' 's:\(<testcase [^>]*status="notrun".*\)/>:\1><skipped/></testcase>:' "$1"
    rm -f "$1.bak"
}


run-single-test-subtests() {
    local test=$1

    # List all the subtests in this test
    bash -c "$build_dir/bin/$test --gtest_list_tests > $output_dir/$test/subtests.tmp.txt"
    IFS=''; while read line; do
        if echo "$line" | grep -q "^  [^ ][^ ]*" ; then
            local current_subtest="$(echo "$line" | sed 's/^  \([^ ][^ ]*\).*/\1/g')"
            echo "$current_test.$current_subtest" >> "$output_dir/$test/subtests.txt"
        else
            local current_test="$(echo "$line" | sed 's/\..*//g')"
        fi
    done < "$output_dir/$test/subtests.tmp.txt"
    rm -f "$output_dir/$test/subtests.tmp.txt"

    # Run the subtests
    printf "\n\nRunning $test subtests\n"
    local i=1;
    while read subtest; do
        local output_file="$output_dir/$test/$subtest/report.xml"
        local test_cmd="$build_dir/bin/$test --gtest_output=xml:$output_file --gtest_filter=$subtest 2>&1"
        mkdir -p "$output_dir/$test/$subtest"
        echo "$test_cmd" >> "$output_dir/$test/$subtest/command.txt"

        if [[ $(uname) = Darwin ]]; then
            if [ -e "/usr/local/bin/gdate" ]; then
                date_nanosec_cmd="/usr/local/bin/gdate +%s%N"
            else
                date_nanosec_cmd="date +%s000000000" # fallback: seconds * 1000000000
            fi
        else
            date_nanosec_cmd="date +%s%N"
        fi

        begin_millisec="$(($(bash -c $date_nanosec_cmd)/1000000))"
        bash -c "$test_cmd" | tee "$output_dir/$test/$subtest/output.txt" ; pipestatus="${PIPESTATUS[0]}"
        end_millisec="$(($(bash -c $date_nanosec_cmd)/1000000))"

        elapsed_millisec="$(($end_millisec - $begin_millisec))"
        elapsed_sec="$(($elapsed_millisec/1000)).$(printf "%03d" $elapsed_millisec)"

        echo "$pipestatus" > "$output_dir/$test/$subtest/status.txt"
        if [ $pipestatus -gt 1 ]; then # this subtest crashed (0:OK 1:failure >1:crash)
            IFS='.' read -r -a array <<< "$subtest"
            test_name="${array[0]}"
            subtest_name="${array[1]}"
            echo "$0: error: $subtest ended with code $pipestatus" >&2
            # Write the XML output by hand
            echo '<?xml version="1.0" encoding="UTF-8"?>
<testsuites tests="1" failures="0" disabled="0" errors="1" time="-'"$elapsed_sec"'" name="AllTests">
    <testsuite name="'"$test_name"'" tests="1" failures="0" disabled="0" errors="1" time="-'"$elapsed_sec"'">
        <testcase name="'"$subtest_name"'" type_param="" status="run" time="-'"$elapsed_sec"'" classname="'"$test_name"'">
            <error message="[CRASH] '"$subtest"' ended with code '"$pipestatus"'">
<![CDATA['"$(cat $output_dir/$test/$subtest/output.txt)"']]>
            </error>
        </testcase>
    </testsuite>
</testsuites>' > "$output_file"
        fi

        if [ -f "$output_file" ]; then
            fix-test-report "$output_file"
            cp "$output_file" "$output_dir/reports/"$test"_subtest"$(printf "%03d" $i)".xml"
        else
            echo "$0: error: $test subtest $subtest ended with code $(cat $output_dir/$test/$subtest/status.txt)" >&2
        fi
        i=$((i + 1))
    done < "$output_dir/$test/subtests.txt"
}

run-single-test() {
    local test=$1
    local output_file="$output_dir/$test/report.xml"
    local test_cmd="$build_dir/bin/$test --gtest_output=xml:$output_file 2>&1"
    # local timeout=900

    echo "$test_cmd" > "$output_dir/$test/command.txt"
    # echo "Running $test with a timeout of $timeout seconds"
    printf "\n\nRunning $test\n"
    rm -f report.xml
    # "$src_dir/scripts/ci/timeout.sh" test "$test_cmd" $timeout | tee $output_dir/$test/output.txt
    bash -c "$test_cmd" | tee $output_dir/$test/output.txt
    status="${PIPESTATUS[0]}"
    echo "$status" > "$output_dir/$test/status.txt"
    # if [[ -e test.timeout ]]; then
    #     echo 'Timeout!'
    #     echo timeout > "$output_dir/$test/status.txt"
    #     rm -f test.timeout
    # else
    #     cat test.exit_code > "$output_dir/$test/status.txt"
    # fi
    # rm -f test.exit_code

    if [ -f "$output_file" ]; then
        if [ "$status" -gt 1 ]; then # report exists but gtest crashed
            echo "$0: fatal: unexpected crash of $test with code $status" >&2
            exit $status
        fi
        fix-test-report "$output_file"
        cp "$output_file" "$output_dir/reports/$test.xml"
    else # no report = some subtest crashed. Let's find out which one.
        echo "$0: error: $test ended with code $status" >&2
        # Run each subtest of this test to avoid results loss
        run-single-test-subtests "$test"
    fi
}

run-all-tests() {
    while read test; do
        run-single-test "$test"
    done < "$output_dir/tests.txt"
}


count-test-suites() {
    list-tests | wc -w | tr -d ' '
}
count-test-reports() {
    ls "$output_dir/reports/" --ignore="*subtest*" 2> /dev/null | wc -l | tr -d ' '
}
count-crashes() {
    echo "$(( $(count-test-suites) - $(count-test-reports) ))"
}

# Fetch the <testsuites> XML elements in reports/*.xml,
# extract and sum the attribute given in argument
# This function relies on the element being written on a single line:
# E.g. <testsuites tests="212" failures="4" disabled="0" errors="0" ...
tests-get()
{
    # Check the existence of report files
    if ! ls "$output_dir/reports/"*.xml &> /dev/null; then
        echo 0
        return
    fi
    attribute="$1"

    # grep the lines containing '<testsuites'; for each one, match the
    # 'attribute="..."' pattern, and collect the "..." part
    counts=$(sed -ne "s/.*<testsuites[^>]* $attribute=\"//" \
                 -e "/^[0-9]/s/\".*//p" "$output_dir/reports/"*.xml)
    # sum the values
    total=0
    for value in $counts; do
        total=$(( $total + $value ))
    done
    echo "$total"
}

print-summary() {
    echo "Testing summary:"
    echo "- $(count-test-suites) test suite(s)"
    echo "- $(tests-get tests) test(s)"
    echo "- $(tests-get disabled) disabled test(s)"
    echo "- $(tests-get failures) failure(s)"

    local errors="$(tests-get errors)"
    echo "- $errors error(s)"
    if [[ "$errors" != 0 ]]; then
        while read test; do
            if [[ ! -e "$output_dir/$test/report.xml" ]]; then # this test crashed
                local status="$(cat "$output_dir/$test/status.txt")"
                case "$status" in
                    "timeout")
                        echo "  - Timeout: $test"
                        ;;
                    [0-9]*)
                        if [[ "$status" -gt 128 && ( $(uname) = Darwin || $(uname) = Linux ) ]]; then
                            echo "  - Exit with status $status ($(kill -l $status)): $test"
                        elif [[ "$status" != 0 ]]; then
                            echo "  - Exit with status $status: $test"
                        fi
                        ;;
                    *)
                        echo "Error: unexpected value in $output_dir/$test/status.txt: $status"
                        ;;
                esac
            fi
        done < "$output_dir/tests.txt"
    fi
}

if [[ "$command" = run ]]; then
    initialize-testing
    run-all-tests
elif [[ "$command" = count-tests ]]; then
    tests-get tests
elif [[ "$command" = count-failures ]]; then
    tests-get failures
elif [[ "$command" = count-disabled ]]; then
    tests-get disabled
elif [[ "$command" = count-errors ]]; then
    tests-get errors
elif [[ "$command" = count-test-suites ]]; then
    count-test-suites
elif [[ "$command" = count-crashes ]]; then
    count-crashes
elif [[ "$command" = print-summary ]]; then
    print-summary
else
    usage
fi
