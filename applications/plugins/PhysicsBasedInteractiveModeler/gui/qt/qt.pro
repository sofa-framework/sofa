# Target is a library: pimguiqt

SOFA_DIR = ../../../../..
TEMPLATE = lib
include($${SOFA_DIR}/sofa.cfg)

DEFINES += SOFA_BUILD_PIMGUIQT

TARGET = pimguiqt$$LIBSUFFIX

CONFIG += $$CONFIGLIBRARIES qt
CONFIG -= staticlib
CONFIG += dll

LIBS += -lpimgui$$LIBSUFFIX
LIBS += -lsofacomponentcollision$$LIBSUFFIX
LIBS += -lPhysicsBasedInteractiveModeler

SOURCES = QMouseOperations.cpp

HEADERS = QMouseOperations.h
