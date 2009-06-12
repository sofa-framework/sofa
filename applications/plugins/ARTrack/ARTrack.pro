
######  GENERAL PLUGIN CONFIGURATION, you shouldn't have to modify it

SOFA_DIR=../../..
TEMPLATE = lib

include($${SOFA_DIR}/sofa.cfg)

DESTDIR = $$SOFA_DIR/lib/sofa-plugins

#set configuration to dynamic library
CONFIG += $$CONFIGLIBRARIES
CONFIG -= staticlib
CONFIG += dll

###### SPECIFIC PLUGIN CONFIGURATION, you should modify it to configure your plugin

TARGET = ARTrack$$LIBSUFFIX

LIBS += $$SOFA_LIBS
LIBS += $$SOFA_EXT_LIBS
INCLUDEPATH += $$SOFA_DIR/extlibs

SOURCES = ARTrackDriver.cpp \
          mainTracker.cpp \
          initARTrack.cpp \
          ARTrackEvent.cpp

HEADERS = ARTrackDriver.h \
          dtracklib.h \
          ARTrackEvent.h
