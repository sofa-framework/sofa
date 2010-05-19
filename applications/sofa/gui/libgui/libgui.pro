# Target is a library: sofagui

SOFA_DIR = ../../../..
TEMPLATE = lib
TARGET = sofagui

include($${SOFA_DIR}/sofa.cfg)

DEFINES += SOFA_BUILD_SOFAGUI

CONFIG += $$CONFIGLIBRARIES
!contains(CONFIGSTATIC, static) {
	CONFIG -= staticlib
CONFIG += dll
}

LIBS += $$SOFA_FRAMEWORK_LIBS
LIBS += $$SOFA_MODULES_LIBS
LIBS += $$SOFA_COMPONENTS_LIBS
LIBS += $$SOFA_EXT_LIBS
 


SOURCES = \
	    ../SofaGUI.cpp \
	    ../BatchGUI.cpp \
            ../MouseOperations.cpp \
            ../PickHandler.cpp \
            ../FilesRecentlyOpenedManager.cpp

HEADERS = \
	    ../SofaGUI.h \
	    ../BatchGUI.h \
            ../OperationFactory.h \
            ../MouseOperations.h \
            ../PickHandler.h \
            ../FilesRecentlyOpenedManager.h 

