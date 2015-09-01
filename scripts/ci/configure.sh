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
        # Only change from the default configuration: Enable tests
        append "-DSOFA_BUILD_TESTS=ON"
        append "-DSOFA-PLUGIN_SOFATEST=ON"
        append "-DSOFA-PLUGIN_SOFAPYTHON=ON"
        append "-DSOFA-PLUGIN_SCENECREATOR=ON"
        ;;
    # Build with as many options enabled
    *options*)

        append "-DSOFA_BUILD_TESTS=ON"

        append "-DSOFA_COMPILE_METIS=ON"
        append "-DSOFA_COMPILE_ARTRACK=ON"

        if [[ ( $(uname) = Darwin || $(uname) = Linux ) ]]; then
            append "-DSOFA-MISC_C++11=ON"
        else
            append "-DSOFA-MISC_C++11=OFF"
        fi

        if [[ -n "$CI_QT_PATH" ]]; then
            append "-DQT_ROOT=$CI_QT_PATH"
        fi

        if [[ -n "$CI_HAVE_BOOST" ]]; then
            append "-DBOOST_ROOT=$CI_BOOST_PATH"
        fi

        if [[ -n "$CI_BULLET_DIR" ]]; then
            append "-DBullet_DIR=$CI_BULLET_DIR"
        fi

        # Bug on Ubuntu 14.04:
        #   Linking CXX executable ../../../bin/sofaTypedefs
        #   Inconsistency detected by ld.so: dl-version.c: 224: _dl_check_map_versions: Assertion `needed != ((void *)0)' failed!
        if [[ $(uname) = Linux ]]; then
            append "-DSOFA-APPLICATION_SOFATYPEDEFS=OFF"
        else
            append "-DSOFA-APPLICATION_SOFATYPEDEFS=ON"
        fi
        append "-DSOFA-APPLICATION_SOFAVERIFICATION=ON"
        # ?
        append "-DSOFA-APPLICATION_XMLCONVERT-DISPLAYFLAGS=OFF"

        ### Plugins
        append "-DSOFA-PLUGIN_ARTRACK=ON"
        if [[ -n "$CI_BULLET_DIR" ]]; then
            append "-DSOFA-PLUGIN_BULLETCOLLISIONDETECTION=ON"
        else
            append "-DSOFA-PLUGIN_BULLETCOLLISIONDETECTION=OFF"
        fi
        # Missing CGAL library
        append "-DSOFA-PLUGIN_CGALPLUGIN=OFF"
        # For Windows, there is the dll of the assimp library *inside* the repository
        if [[ ( $(uname) = Darwin || $(uname) = Linux ) && -z "$CI_HAVE_ASSIMP" ]]; then
            append "-DSOFA-PLUGIN_COLLADASCENELOADER=OFF"
        else
            append "-DSOFA-PLUGIN_COLLADASCENELOADER=ON"
        fi
        append "-DSOFA-PLUGIN_COMPLIANT=ON"
        append "-DSOFA-PLUGIN_EXTERNALBEHAVIORMODEL=ON"
        append "-DSOFA-PLUGIN_FLEXIBLE=ON"
        # Requires specific libraries.
        append "-DSOFA-PLUGIN_HAPTION=OFF"
        append "-DSOFA-PLUGIN_IMAGE=ON"
        append "-DSOFA-PLUGIN_INVERTIBLEFVM=ON"
        append "-DSOFA-PLUGIN_MANIFOLDTOPOLOGIES=ON"
        append "-DSOFA-PLUGIN_MANUALMAPPING=ON"
        if [ -n "$CI_HAVE_OPENCASCADE" ]; then
            append "-DSOFA-PLUGIN_MESHSTEPLOADER=ON"
        else
            append "-DSOFA-PLUGIN_MESHSTEPLOADER=OFF"
        fi
        append "-DSOFA-PLUGIN_MULTITHREADING=ON"
        append "-DSOFA-PLUGIN_OPTITRACKNATNET=ON"
        # Does not compile, but it just needs to be updated.
        append "-DSOFA-PLUGIN_PERSISTENTCONTACT=OFF"
        append "-DSOFA-PLUGIN_PLUGINEXAMPLE=ON"
        append "-DSOFA-PLUGIN_REGISTRATION=ON"
        append "-DSOFA-PLUGIN_SCENECREATOR=ON"
        # Requires OpenHaptics libraries.
        append "-DSOFA-PLUGIN_SENSABLE=OFF"
        if [[ -n "$CI_HAVE_BOOST" ]]; then
            append "-DSOFA-PLUGIN_SENSABLEEMULATION=ON"
        else
            append "-DSOFA-PLUGIN_SENSABLEEMULATION=OFF"
        fi
        # Requires Sixense libraries.
        append "-DSOFA-PLUGIN_SIXENSEHYDRA=OFF"
        append "-DSOFA-PLUGIN_SOFACARVING=ON"
        if [[ -n "$CI_HAVE_CUDA" ]]; then
            append "-DSOFA-PLUGIN_SOFACUDA=ON"
        else
            append "-DSOFA-PLUGIN_SOFACUDA=OFF"
        fi
        # Requires HAPI libraries.
        append "-DSOFA-PLUGIN_SOFAHAPI=OFF"
        # Not sure if worth maintaining
        append "-DSOFA-PLUGIN_SOFAPML=OFF"
        append "-DSOFA-PLUGIN_SOFASIMPLEGUI=ON"
        append "-DSOFA-PLUGIN_SOFATEST=ON"
        if [[ -n "$CI_HAVE_BOOST" ]]; then
            if [[ $(uname) = Linux ]]; then
                # Currently compiles only on Linux...
                append "-DSOFA-PLUGIN_THMPGSPATIALHASHING=ON"
            else
                append "-DSOFA-PLUGIN_THMPGSPATIALHASHING=OFF"
            fi
        else
            append "-DSOFA-PLUGIN_THMPGSPATIALHASHING=OFF"
        fi
        # Requires SofaCUDALDI (Strange, SofaCUDALDI is in sofa-dev!)
        append "-DSOFA-PLUGIN_VOXELIZER=OFF"
        # Requires XiRobot library.
        append "-DSOFA-PLUGIN_XITACT=OFF"
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
