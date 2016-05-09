#!/bin/bash

# This script creates the build directory if it does not exist.  If the build
# already exists, the script checks if it is possible to make an incremental
# build.

## Significant environnement variables:
# - CI_FORCE_FULL_BUILD       Prevent an incremental build

# Exit on error
set -o errexit


### Checks

usage() {
    echo "Usage: init.sh <build-dir> <src-dir>"
}

if [[ "$#" = 2 ]]; then
    build_dir="$1"
    src_dir="$(cd "$2" && pwd)"
else
    usage; exit 1
fi

if [[ ! -d "$src_dir/applications/plugins" ]]; then
    echo "Error: '$src_dir' does not look like a Sofa source tree."
    usage; exit 1
fi

if [[ ! -d "$build_dir" ]]; then
    mkdir -p "$build_dir"
fi
build_dir="$(cd "$build_dir" && pwd)"


### Actual work

sha=$(git --git-dir="$src_dir/.git" rev-parse HEAD)

## Check if an incremental build is possible

full_build=""
if [ ! -z "$CI_FORCE_FULL_BUILD" ]; then
    full_build="Full build forced."
elif [ ! -e "$build_dir/CMakeCache.txt" ]; then
    full_build="No previous build detected."
elif [ ! -e "$build_dir/last-commit-built.txt" ]; then
    full_build="Last build's commit not found."
else
    # Sometimes, a change in a cmake script can cause an incremental
    # build to fail, so let's be extra cautious and make a full build
    # each time a .cmake file changes.
    last_commit_build="$(cat "$build_dir/last-commit-built.txt")"
    if git --git-dir="$src_dir/.git" diff --name-only "$last_commit_build" "$sha" | grep 'cmake/.*\.cmake' ; then
        full_build="Detected changes in a CMake script file."
    fi
fi


if [ -n "$full_build" ]; then
    echo "Starting a full build. ($full_build)"
    # '|| true' is an ugly workaround, because rm sometimes fails to remove the
    # build directory on the Windows slaves, for reasons unknown yet.
    rm -rf "$build_dir" || true
    mkdir -p "$build_dir"
    # Flag. E.g. we check this before counting compiler warnings,
    # which is not relevant after an incremental build.
    touch "$build_dir/full-build"
    echo "$sha" > "$build_dir/last-commit-built.txt"
else
    rm -f "$build_dir/full-build"
    echo "Starting an incremental build"
fi
