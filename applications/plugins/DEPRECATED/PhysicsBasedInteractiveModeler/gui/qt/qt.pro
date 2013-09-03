# Target is a library: pimguiqt

SOFA_DIR = ../../../../..
TEMPLATE = lib
include($${SOFA_DIR}/sofa.cfg)

DEFINES += SOFA_BUILD_PIMGUIQT

TARGET = pimguiqt$$LIBSUFFIX

CONFIG += $$CONFIGLIBRARIES qt
!contains(CONFIGSTATIC, static) {
	CONFIG -= staticlib
CONFIG += dll
}
QT += opengl qt3support xml

SOFA_CGAL_PATH = $$SOFA_DIR/extlibs/CGAL
INCLUDEPATH += $$SOFA_CGAL_PATH/include
DEPENDPATH += $$SOFA_CGAL_PATH/include

LIBS += -lpimgui$$LIBSUFFIX
LIBS += -lsofacomponentcollision$$LIBSUFFIX
LIBS += -lpim

SOURCES = QMouseOperations.cpp

HEADERS = QMouseOperations.h
