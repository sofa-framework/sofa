#!/bin/bash

# Here we pick what gets to be compiled. The role of this script is to
# call cmake with the appropriate options. After this, the build
# directory should be ready to run 'make'.

## Significant environnement variables:
# - CI_JOB                    (e.g. ubuntu_gcc-4.8_options)
# - CI_CMAKE_OPTIONS          (additional arguments to pass to cmake)
# - CI_ARCH = x86 | amd64     (for Windows builds)
# - CI_BUILD_TYPE             Debug|Release
# - CC and CXX
# About available libraries:
# - CI_HAVE_BOOST
# - CI_BOOST_PATH             (empty string if installed in standard location)
# - CI_QT_PATH
# - CI_BULLET_DIR             (Path to the directory containing BulletConfig.cmake)
# - CI_HAVE_ASSIMP
# - CI_HAVE_OPENCASCADE
# - CI_HAVE_CUDA
# - CI_HAVE_OPENCL


# Exit on error
set -o errexit


## Checks

usage() {
    echo "Usage: configure.sh <build-dir> <src-dir>"
}

if [[ "$#" = 2 ]]; then
    build_dir="$1"
    if [[ $(uname) = Darwin || $(uname) = Linux ]]; then
        src_dir="$(cd "$2" && pwd)"
    else
        # pwd with a Windows format (c:/ instead of /c/)
        src_dir="$(cd "$2" && pwd -W)"
    fi
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


## Defaults

if [ -z "$CI_ARCH" ]; then CI_ARCH="x86"; fi
if [ -z "$CI_JOB" ]; then CI_JOB="$JOB_NAME"; fi
if [ -z "$CI_JOB" ]; then CI_JOB="default"; fi
if [ -z "$CI_BUILD_TYPE" ]; then CI_BUILD_TYPE="Release"; fi


## Utils

generator() {
    if [[ $(uname) = Darwin || $(uname) = Linux ]]; then
        echo "Unix Makefiles"
    else
        echo "\"NMake Makefiles\""
    fi
}

call-cmake() {
    if [ $(uname) != Darwin -a $(uname) != Linux ]; then
        # Run cmake after calling vcvarsall.bat to setup compiler stuff
        local vcvarsall="call \"%VS110COMNTOOLS%..\\..\\VC\vcvarsall.bat\" $CI_ARCH"
        echo "Calling $COMSPEC /c \"$vcvarsall & cmake $*\""
        $COMSPEC /c "$vcvarsall & cmake $*"
    else
        cmake "$@"
    fi
}


## CMake options

cmake_options="-DCMAKE_BUILD_TYPE=$CI_BUILD_TYPE"

append() {
    cmake_options="$cmake_options $*"
}

case $CI_JOB in
    # Build with default options
    *default*)
        ;;
    # Build with as many options enabled
    *options*)
        ;;
esac

# Options passed via the environnement
if [ ! -z "$CI_CMAKE_OPTIONS" ]; then
    cmake_options="$cmake_options $CI_CMAKE_OPTIONS"
fi

cd "$build_dir"


## Preconfigure

if [ -e "full-build" ]; then
    call-cmake -G"$(generator)" "$src_dir"
fi


## Configure

echo "Calling cmake with the following options:"
echo "$cmake_options" | tr -s ' ' '\n'
call-cmake $cmake_options .

# Work around a bug in the cmake scripts, where the include directories of gtest
# are not searched after the first "configure".
if [ -e "full-build" ]; then
    echo "Calling cmake again, to workaround the gtest missing include directories bug."
    call-cmake .
fi
