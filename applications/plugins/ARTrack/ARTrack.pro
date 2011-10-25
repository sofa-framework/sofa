######  PLUGIN TARGET
TARGET = ARTrackPlugin

######  GENERAL PLUGIN CONFIGURATION, you shouldn't have to modify it

SOFA_DIR=../../..
TEMPLATE = lib

include($${SOFA_DIR}/sofa.cfg)

DESTDIR = $$SOFA_DIR/lib/sofa-plugins

#set configuration to dynamic library
CONFIG += $$CONFIGLIBRARIES
!contains(CONFIGSTATIC, static) {
	CONFIG -= staticlib
CONFIG += dll
}

###### SPECIFIC PLUGIN CONFIGURATION, you should modify it to configure your plugin

DEFINES += SOFA_BUILD_ARTRACKPLUGIN

LIBS += $$SOFA_LIBS
SOFA_EXT_LIBS += -lARTracklib$$LIBSUFFIX
LIBS += $$SOFA_EXT_LIBS
INCLUDEPATH += $$SOFA_DIR/extlibs/ARTrack
DEPENDPATH += $$SOFA_DIR/extlibs/ARTrack

SOURCES = \
ARTrackDriver.cpp \
          initARTrack.cpp \
          ARTrackEvent.cpp \
          ARTrackController.cpp

HEADERS = \
ARTrackDriver.h \
          ARTrackEvent.h \
          ARTrackController.h \
          ARTrackController.inl
