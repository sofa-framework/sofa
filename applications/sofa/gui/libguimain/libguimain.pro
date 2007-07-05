# Target is a library: sofagui

SOFA_DIR = ../../../..
TEMPLATE = lib
include($$SOFA_DIR/sofa.cfg)

TARGET = sofaguimain$$LIBSUFFIX
CONFIG += $$CONFIGLIBRARIES
LIBS += $$SOFA_FRAMEWORK_LIBS $$SOFA_MODULES_LIBS
LIBS += -lsofagui$$LIBSUFFIX
LIBS -= -lsofaguimain$$LIBSUFFIX # remove ourself from the list of libs
LIBS += $$SOFA_EXT_LIBS

SOURCES = \
	    ../Main.cpp

HEADERS = 

