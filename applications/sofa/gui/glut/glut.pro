# Target is a library: sofaguiglut

SOFA_DIR = ../../../..
TEMPLATE = lib
TARGET = sofaguiglut

include($${SOFA_DIR}/sofa.cfg)

DEFINES += SOFA_BUILD_SOFAGUIGLUT

CONFIG += $$CONFIGLIBRARIES
!contains(CONFIGSTATIC, static) {
	CONFIG -= staticlib
CONFIG += dll
}

LIBS += -lsofagui$$LIBSUFFIX
LIBS += $$SOFA_FRAMEWORK_LIBS $$SOFA_MODULES_LIBS 
LIBS += $$SOFA_EXT_LIBS
SOURCES = \
	    SimpleGUI.cpp

HEADERS = \
	    SimpleGUI.h

