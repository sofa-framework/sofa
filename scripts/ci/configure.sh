#!/bin/bash

# Here we pick what gets to be compiled. The role of this script is to
# call cmake with the appropriate options. After this, the build
# directory should be ready to run 'make'.

## Significant environnement variables:
# - CI_JOB                    (e.g. ubuntu_gcc-4.8_options)
# - CI_OPTIONS                if contains "options" then activate plugins
# - CI_CMAKE_OPTIONS          (additional arguments to pass to cmake)
# - CI_ARCH = x86 | amd64     (for Windows builds)
# - CI_BUILD_TYPE             Debug|Release
# - CC and CXX
# - CI_COMPILER               # important for Visual Studio paths (VS-2012, VS-2013 or VS-2015)
# About available libraries:
# - CI_HAVE_BOOST
# - CI_BOOST_PATH             (empty string if installed in standard location)
# - CI_QT_PATH
# - CI_BULLET_DIR             (Path to the directory containing BulletConfig.cmake)
# - CI_HAVE_ASSIMP
# - CI_HAVE_OPENCASCADE
# - CI_HAVE_CUDA
# - CI_HAVE_OPENCL
# - CI_HAVE_CSPARSE
# - CI_HAVE_METIS


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
    if [ -x "$(command -v ninja)" ]; then
        echo "Ninja"
    elif [[ $(uname) = Darwin || $(uname) = Linux ]]; then
        echo "Unix Makefiles"
    else
        echo "\"NMake Makefiles\""
    fi
}

call-cmake() {
    pwd
    if [[ "$(uname)" != "Darwin" && "$(uname)" != "Linux" ]]; then
        # Call vcvarsall.bat first to setup environment
        if [ "$CI_COMPILER" = "VS-2015" ]; then
            vcvarsall="call \"%VS140COMNTOOLS%..\\..\\VC\vcvarsall.bat\" $CI_ARCH"
        elif [ "$CI_COMPILER" = "VS-2013" ]; then
            vcvarsall="call \"%VS120COMNTOOLS%..\\..\\VC\vcvarsall.bat\" $CI_ARCH"
        else
            vcvarsall="call \"%VS110COMNTOOLS%..\\..\\VC\vcvarsall.bat\" $CI_ARCH"
        fi
        echo "Calling $COMSPEC /c \"$vcvarsall & cmake $*\""
        $COMSPEC /c "$vcvarsall & cmake $*"
    else
        cmake "$@"
    fi
}


## CMake options

cmake_options="-DCMAKE_COLOR_MAKEFILE=OFF -DCMAKE_BUILD_TYPE=$CI_BUILD_TYPE"

append() {
    cmake_options="$cmake_options $*"
}

# Options common to all configurations
append "-DSOFA_WITH_DEPRECATED_COMPONENTS=ON"
append "-DAPPLICATION_GETDEPRECATEDCOMPONENTS=ON"
append "-DSOFA_BUILD_TUTORIALS=ON"
append "-DSOFA_BUILD_TESTS=ON"
append "-DSOFAGUI_BUILD_TESTS=OFF"
append "-DPLUGIN_SOFAPYTHON=ON"
if [[ -n "$CI_HAVE_BOOST" ]]; then
    append "-DBOOST_ROOT=$CI_BOOST_PATH"
fi

case $CI_OPTIONS in
    # Build with as many options enabled as possible
    *options*)
        append "-DSOFA_BUILD_METIS=ON"
        append "-DSOFA_BUILD_ARTRACK=ON"
        append "-DSOFA_BUILD_MINIFLOWVR=ON"

        if [[ -n "$CI_QT_PATH" ]]; then
            append "-DQT_ROOT=$CI_QT_PATH"
        fi

        if [[ -n "$CI_BULLET_DIR" ]]; then
            append "-DBullet_DIR=$CI_BULLET_DIR"
        fi
        
        # HeadlessRecorder is Linux only for now
        if [[ $(uname) = Linux ]]; then
        id=$(cat /etc/*-release | grep "ID")
        if [[ $id = *"centos"* ]]; then
            append "-DSOFAGUI_HEADLESS_RECORDER=OFF"
        else
            append "-DSOFAGUI_HEADLESS_RECORDER=ON"
        fi
        fi

        ### Plugins
        append "-DPLUGIN_ARTRACK=ON"
        if [[ -n "$CI_BULLET_DIR" ]]; then
            append "-DPLUGIN_BULLETCOLLISIONDETECTION=ON"
        else
            append "-DPLUGIN_BULLETCOLLISIONDETECTION=OFF"
        fi
        if [[ -n "$CI_HAVE_CGAL" ]]; then
            append "-DPLUGIN_CGALPLUGIN=ON"
        else
            append "-DPLUGIN_CGALPLUGIN=OFF"
        fi
        if [[ ( $(uname) = Darwin || $(uname) = Linux ) && -z "$CI_HAVE_ASSIMP" ]]; then
            append "-DPLUGIN_COLLADASCENELOADER=OFF"
        else
            # For Windows, Assimp dll is in the repository
            append "-DPLUGIN_COLLADASCENELOADER=ON"
        fi
        append "-DPLUGIN_COMPLIANT=ON"
        append "-DPLUGIN_EXTERNALBEHAVIORMODEL=ON"
        append "-DPLUGIN_FLEXIBLE=ON"
        append "-DPLUGIN_HAPTION=OFF" # Requires specific libraries.
        append "-DPLUGIN_IMAGE=ON"
        append "-DPLUGIN_INVERTIBLEFVM=ON"
        append "-DPLUGIN_MANIFOLDTOPOLOGIES=ON"
        append "-DPLUGIN_MANUALMAPPING=ON"
        if [ -n "$CI_HAVE_OPENCASCADE" ]; then
            append "-DPLUGIN_MESHSTEPLOADER=ON"
        else
            append "-DPLUGIN_MESHSTEPLOADER=OFF"
        fi
        append "-DPLUGIN_MULTITHREADING=ON"
        append "-DPLUGIN_OPTITRACKNATNET=ON"
        append "-DPLUGIN_PERSISTENTCONTACT=OFF" # Does not compile, but it just needs to be updated.
        append "-DPLUGIN_PLUGINEXAMPLE=ON"
        append "-DPLUGIN_REGISTRATION=ON"
        append "-DPLUGIN_RIGIDSCALE=ON"
        append "-DPLUGIN_SENSABLE=OFF" # Requires OpenHaptics libraries.
        if [[ -n "$CI_HAVE_BOOST" ]]; then
            append "-DPLUGIN_SENSABLEEMULATION=ON"
        else
            append "-DPLUGIN_SENSABLEEMULATION=OFF"
        fi
        append "-DPLUGIN_SIXENSEHYDRA=OFF" # Requires Sixense libraries.
        append "-DPLUGIN_SOFACARVING=ON"
        if [[ -n "$CI_HAVE_CUDA" ]]; then
            append "-DPLUGIN_SOFACUDA=ON"
        else
            append "-DPLUGIN_SOFACUDA=OFF"
        fi
        append "-DPLUGIN_SOFADISTANCEGRID=ON" # Requires MiniFlowVR for DistanceGridForceField-liver.scn
        append "-DPLUGIN_SOFAEULERIANFLUID=ON"
        append "-DPLUGIN_SOFAHAPI=OFF" # Requires HAPI libraries.
        append "-DPLUGIN_SOFAIMPLICITFIELD=ON"
        append "-DPLUGIN_SOFAMISCCOLLISION=ON"
        append "-DPLUGIN_SOFASIMPLEGUI=ON" # Not sure if worth maintaining
        append "-DPLUGIN_SOFASPHFLUID=ON"
        append "-DPLUGIN_THMPGSPATIALHASHING=ON"
        append "-DPLUGIN_XITACT=OFF" # Requires XiRobot library.
        ;;
esac

# Options passed via the environnement
if [ ! -z "$CI_CMAKE_OPTIONS" ]; then
    cmake_options="$cmake_options $CI_CMAKE_OPTIONS"
fi

cd "$build_dir"

## Configure

echo "Calling cmake with the following options:"
echo "$cmake_options" | tr -s ' ' '\n'
if [ -e "full-build" ]; then
    call-cmake -G"$(generator)" $cmake_options "$src_dir"
else
    call-cmake $cmake_options .
fi
