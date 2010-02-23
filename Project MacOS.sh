#! /bin/sh

# generate a XCODE Project
qmake -recursive -spec macx-xcode Sofa.pro

# Generate a Makefile
#qmake -recursive -spec macx-g++ Makefile Sofa.pro
qmake -recursive -spec macx-g++
