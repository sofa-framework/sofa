# Target is a library: sofagui

SOFA_DIR = ../../../..
TEMPLATE = lib
include($${SOFA_DIR}/sofa.cfg)

TARGET = sofaguimain$$LIBSUFFIX

DEFINES += SOFA_BUILD_GUIMANAGER
contains (DEFINES, SOFA_QT4) {
	CONFIG += $$CONFIGLIBRARIES qt 
}
else{
	CONFIG += $$CONFIGLIBRARIRES qt
}
CONFIG -= staticlib
CONFIG += dll

LIBS += $$SOFA_FRAMEWORK_LIBS $$SOFA_MODULES_LIBS $$SOFA_GUI_LIBS
LIBS -= -lsofaguimain$$LIBSUFFIX # remove ourself from the list of libs
LIBS += $$SOFA_GUI_EXT_LIBS $$SOFA_EXT_LIBS

SOURCES = \
	    ../GUIManager.cpp

HEADERS = \ 
		../GUIManager.h

