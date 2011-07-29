#!/bin/bash
PLATFORM=$(uname)
if [ "$PLATFORM" = "Darwin" ]; then
PLATFORM=macx
else
PLATFORM=linux
fi
export SOFA_DIR=$PWD
export PATH=$SOFA_DIR/bin:$PATH
export LD_LIBRARY_PATH=$SOFA_DIR/lib:$LD_LIBRARY_PATH
if [ "$PLATFORM" = "macx" ]; then
export DYLD_LIBRARY_PATH=$SOFA_DIR/lib:$DYLD_LIBRARY_PATH
fi
