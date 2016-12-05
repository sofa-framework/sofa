#!/bin/bash


# Messy main script that glues the other ones together.


# Exit on error
set -o errexit

build_dir="$1"
src_dir="$(cd "$2" && pwd)"
sha=$(git --git-dir="$src_dir/.git" rev-parse HEAD)

# Clean flag files
rm -f "$build_dir/build-started"
rm -f "$build_dir/build-finished"

if [ -z "$CI_JOB" ]; then CI_JOB="$JOB_NAME"; fi

send-message-to-dashboard() {
    if [ -z "$CI_DASHBOARD_URL" ]; then
        echo "Message (not sent): " sha="$sha" "config=$CI_JOB" $*
        true
    else
        local message="$1"
        while [ $# -gt 1 ]; do
            shift
            message="$message&$1"
        done
        message="$message&sha=$sha&config=$CI_JOB"
        local url="$CI_DASHBOARD_URL"
        echo "Message (sent): " sha="$sha" "config=$CI_JOB" $*
        wget --no-verbose --output-document=/dev/null --post-data="$message" "$CI_DASHBOARD_URL"
    fi
}

notify-failure() {
    local exit_code=$?
    echo "Error detected, build aborted."
    touch "$build_dir/build-finished"
    send-message-to-dashboard "status=fail"
    exit $exit_code
}

"$src_dir/scripts/ci/init-build.sh" "$build_dir" "$src_dir"

touch "$build_dir/build-started"

send-message-to-dashboard "platform=$CI_PLATFORM" "compiler=$CI_COMPILER" "options=$CI_OPTIONS" "build_url=$BUILD_URL" "job_url=$JOB_URL"
send-message-to-dashboard "status=build"
trap notify-failure ERR

## Configure

"$src_dir/scripts/ci/configure.sh" "$build_dir" "$src_dir"

## Compile

"$src_dir/scripts/ci/compile.sh" "$build_dir"

## Run the tests

"$src_dir/scripts/ci/tests.sh" run "$build_dir" "$src_dir"

send-message-to-dashboard \
    "tests_total=$("$src_dir/scripts/ci/tests.sh" count-tests $build_dir $src_dir)" \
    "tests_failures=$("$src_dir/scripts/ci/tests.sh" count-failures $build_dir $src_dir)" \
    "tests_disabled=$("$src_dir/scripts/ci/tests.sh" count-disabled $build_dir $src_dir)" \
    "tests_errors=$("$src_dir/scripts/ci/tests.sh" count-errors $build_dir $src_dir)" \
    "tests_suites=$("$src_dir/scripts/ci/tests.sh" count-test-suites $build_dir $src_dir)" \
    "tests_crash=$("$src_dir/scripts/ci/tests.sh" count-crashes $build_dir $src_dir)"

touch "$build_dir/build-finished"

## Count Warnings

count_warnings() {
    local warning_count=-1
    if [[ $(uname) = Darwin || $(uname) = Linux ]]; then
        warning_count=$(grep '^[^:]\+:[0-9]\+:[0-9]\+: warning:' "$build_dir/make-output.txt" | sort -u | wc -l | tr -d ' ')
    else
        warning_count=$(grep ' : warning [A-Z]\+[0-9]\+:' "$build_dir/make-output.txt" | sort | uniq | wc -l)
    fi
    echo "$warning_count"
}

if [ -e "$build_dir/full-build" ]; then
    warning_count=$(count_warnings)
    echo "Counted $warning_count compiler warnings."
    send-message-to-dashboard "fullbuild=true" "warnings=$warning_count"
fi

## Test scenes

if [[ -n "$CI_TEST_SCENES" ]]; then
    "$src_dir/scripts/ci/scene-tests.sh" run "$build_dir" "$src_dir"
    scenes_errors_count=$("$src_dir/scripts/ci/scene-tests.sh" count-errors "$build_dir" "$src_dir")
    send-message-to-dashboard "scenes_errors=$scenes_errors_count"
    scenes_crashes_count=$("$src_dir/scripts/ci/scene-tests.sh" count-crashes "$build_dir" "$src_dir")
    send-message-to-dashboard "scenes_crashes=$scenes_crashes_count"
fi

"$src_dir/scripts/ci/tests.sh" print-summary "$build_dir" "$src_dir"
if [[ -n "$CI_TEST_SCENES" ]]; then
    "$src_dir/scripts/ci/scene-tests.sh" print-summary "$build_dir" "$src_dir"
fi

send-message-to-dashboard "status=success"
