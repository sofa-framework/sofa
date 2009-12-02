# Target is a library: pimgui

SOFA_DIR = ../../../../..
TEMPLATE = lib
include($${SOFA_DIR}/sofa.cfg)

DEFINES += SOFA_BUILD_PIMGUI

TARGET = pimgui$$LIBSUFFIX

CONFIG += $$CONFIGLIBRARIES
CONFIG -= staticlib
CONFIG += dll

LIBS += -lsofagui$$LIBSUFFIX
LIBS += -lsofacomponentcollision$$LIBSUFFIX
LIBS += -lpim

SOURCES = ../MouseOperations.cpp

HEADERS = ../MouseOperations.h
