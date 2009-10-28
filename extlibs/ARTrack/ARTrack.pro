# Target is a library:  ARTrack

SOFA_DIR = ../..
TEMPLATE = lib
TARGET = ARTrack

include($${SOFA_DIR}/sofa.cfg)

CONFIG += $$CONFIGLIBRARIES

HEADERS += \
  dtracklib.h

SOURCES += \
  mainTracker.cpp
