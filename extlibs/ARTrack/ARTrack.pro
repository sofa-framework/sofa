# Target is a library:  ARTrack

SOFA_DIR = ../..
TEMPLATE = lib
include($${SOFA_DIR}/sofa.cfg)

TARGET = ARTracklib$$LIBSUFFIX
CONFIG += $$CONFIGLIBRARIES

unix {
    CONFIG *= opengl
    LIBS += -lglut
}

HEADERS += \
  dtracklib.h

SOURCES += \
  mainTracker.cpp
