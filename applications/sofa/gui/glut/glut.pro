# Target is a library: sofaguiglut

SOFA_DIR = ../../../..
TEMPLATE = lib
include($${SOFA_DIR}/sofa.cfg)

TARGET = sofaguiglut$$LIBSUFFIX
CONFIG += $$CONFIGLIBRARIES
LIBS += $$SOFA_FRAMEWORK_LIBS $$SOFA_MODULES_LIBS
LIBS += -lsofagui$$LIBSUFFIX
LIBS += $$SOFA_EXT_LIBS

SOURCES = \
	    SimpleGUI.cpp

HEADERS = \
	    SimpleGUI.h

