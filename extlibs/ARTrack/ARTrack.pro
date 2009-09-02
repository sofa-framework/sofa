# Target is a library:  ARTrack

SOFA_DIR = ../..
TEMPLATE = lib
include($${SOFA_DIR}/sofa.cfg)

TARGET = ARTrack$$LIBSUFFIX
CONFIG += $$CONFIGLIBRARIES

HEADERS += \
  dtracklib.h

SOURCES += \
  mainTracker.cpp
