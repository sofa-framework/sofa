#!/bin/sh
#

SOFA_BUNDLE="`echo "$0" | sed -e 's/\/Contents\/MacOS\/Sofa//'`"
SOFA_RESOURCES="$SOFA_BUNDLE/Contents/Resources"
SOFA_BIN_DIR="$SOFA_RESOURCES/bin/"
SOFA_LIB_DIR="$SOFA_RESOURCES/lib/"
SOFA_EXEC="runSofa"

echo "SOFA_BUNDLE: $SOFA_BUNDLE" 
echo "SOFA_BIN_DIR: $SOFA_BIN_DIR"
echo "SOFA_LIB_DIR: $SOFA_LIB_DIR"
echo "SOFA_LIB_DIR: $SOFA_LIB_DIR"
export DYLD_LIBRARY_PATH=$SOFA_LIB_DIR


cd $SOFA_BIN_DIR && ./$SOFA_EXEC -g qt
 