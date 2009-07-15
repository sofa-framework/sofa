# Target is a library: sofagui

SOFA_DIR = ../../../..
TEMPLATE = lib
include($${SOFA_DIR}/sofa.cfg)

TARGET = sofagui$$LIBSUFFIX
CONFIG += $$CONFIGLIBRARIES
LIBS += $$SOFA_FRAMEWORK_LIBS $$SOFA_MODULES_LIBS
LIBS += $$SOFA_EXT_LIBS

SOURCES = \
	    ../SofaGUI.cpp \
	    ../BatchGUI.cpp \
            ../MouseOperations.cpp \
            ../PickHandler.cpp

HEADERS = \
	    ../SofaGUI.h \
	    ../BatchGUI.h \
            ../OperationFactory.h \
            ../MouseOperations.h \
            ../PickHandler.h

