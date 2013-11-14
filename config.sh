#!/bin/bash

####
# Determine the location of the script.
# This algorithm also works when sourcing the script
# or when some directories are symbolic links.
SCRIPT_PATH="${BASH_SOURCE[0]}";
if([ -h "${SCRIPT_PATH}" ]) then
  while([ -h "${SCRIPT_PATH}" ]) do SCRIPT_PATH=`readlink "${SCRIPT_PATH}"`; done
fi
pushd . > /dev/null
cd `dirname ${SCRIPT_PATH}` > /dev/null
SCRIPT_PATH=`pwd`;
popd  > /dev/null
####

export SRC_DIR=$SCRIPT_PATH
# Add the first param (if it exist) to SCRIPT_PATH
# In general $1 is the cmake build directory
# readlink Remove unnecessary slashes (if $1 is empty)
export BUILD_DIR=$(readlink -m $PWD"/"$1"/")

PLATFORM=$(uname)
if [ "$PLATFORM" = "Darwin" ]; then
PLATFORM=macx
else
PLATFORM=linux
fi

export PATH=$BUILD_DIR/bin:$PATH
export LD_LIBRARY_PATH=$BUILD_DIR/lib:$SRC_DIR/lib/$PLATFORM:$LD_LIBRARY_PATH
if [ "$PLATFORM" = "macx" ]; then
  export DYLD_LIBRARY_PATH=$BUILD_DIR/lib:$SRC_DIR/lib/$PLATFORM:$DYLD_LIBRARY_PATH
fi
export LIBRARY_PATH=$BUILD_DIR/lib:$SRC_DIR/lib/$PLATFORM:$LIBRARY_PATH
export CPATH=$SRC_DIR/include:$CPATH

if [ "$BUILD_DIR" != "$SRC_DIR" ]; then
  export SOFA_DATA_PATH=$SRC_DIR/examples:$SRC_DIR/share:$SOFA_DATA_PATH
fi
