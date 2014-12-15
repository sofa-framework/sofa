#!/bin/bash

# This script basically runs 'make' and saves the compilation output
# in make-output.txt.

## Significant environnement variables:
# - CI_MAKE_OPTIONS       # additional arguments to pass to make
# - CI_ARCH               # x86|amd64  (32-bit or 64-bit build - Windows-specific)

# Exit on error
set -o errexit


### Checks

usage() {
    echo "Usage: compile.sh <build-dir>"
}

if [[ "$#" = 1 ]]; then
    build_dir="$(cd "$1" && pwd)"
else
    usage; exit 1
fi

if [[ ! -e "$build_dir/CMakeCache.txt" ]]; then
    echo "Error: '$build_dir' does not look like a build directory."
    usage; exit 1
fi


### Defaults

if [ -z "$CI_ARCH" ]; then CI_ARCH="x86"; fi


### Actual work

call-make() {
    if [[ "$(uname)" != "Darwin" && "$(uname)" != "Linux" ]]; then
        # We need to call vcvarsall.bat before nmake to setup compiler stuff
        local vcvarsall="call \"%VS110COMNTOOLS%..\\..\\VC\vcvarsall.bat\" $CI_ARCH"
        $COMSPEC /c "$vcvarsall & nmake $CI_MAKE_OPTIONS"
    else
        make $CI_MAKE_OPTIONS
    fi
}

cd "$build_dir"

# The output of make is saved to a file, to check for warnings later. Since make
# is inside a pipe, errors will go undetected, thus we create a file
# 'make-failed' when make fails, to check for errors.
rm -f make-failed
( call-make 2>&1 || touch make-failed ) | tee make-output.txt

if [ -e make-failed ]; then
    exit 1
fi
