# Target is a library: pimgui

SOFA_DIR = ../../../../..
TEMPLATE = lib
include($${SOFA_DIR}/sofa.cfg)

DEFINES += SOFA_BUILD_PIMGUI

TARGET = pimgui$$LIBSUFFIX

CONFIG += $$CONFIGLIBRARIES
!contains(CONFIGSTATIC, static) {
	CONFIG -= staticlib
CONFIG += dll
}

SOFA_CGAL_PATH = $$SOFA_DIR/extlibs/CGAL
INCLUDEPATH += $$SOFA_CGAL_PATH/include
DEPENDPATH += $$SOFA_CGAL_PATH/include

LIBS += -lsofagui$$LIBSUFFIX
LIBS += -lsofacomponentcollision$$LIBSUFFIX
LIBS += -lpim

SOURCES = ../MouseOperations.cpp

HEADERS = ../MouseOperations.h
