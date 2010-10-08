######  PLUGIN TARGET
TARGET = MeshSTEPLoaderPlugin

######  GENERAL PLUGIN CONFIGURATION, you shouldn't have to modify it

SOFA_DIR=../../..
TEMPLATE = lib

include($${SOFA_DIR}/sofa.cfg)

DESTDIR = $$SOFA_DIR/lib/sofa-plugins

#set configuration to dynamic library
CONFIG += $$CONFIGLIBRARIES
CONFIG -= staticlib
CONFIG += dll

###### SPECIFIC PLUGIN CONFIGURATION, you should modify it to configure your plugin

TARGET = MeshSTEPLoaderPlugin$$LIBSUFFIX
DEFINES += SOFA_BUILD_MESHSTEPLOADERPLUGIN

LIBS += $$SOFA_LIBS
LIBS += $$SOFA_EXT_LIBS

LIBS += -l$$OPEN_CASCADE_DIR/win32/lib/*

INCLUDEPATH += $$SOFA_DIR/extlibs
INCLUDEPATH += $$OPEN_CASCADE_DIR/inc

SOURCES = \
MeshSTEPLoader.cpp \
          initMeshSTEPLoader.cpp

HEADERS = \
MeshSTEPLoader.h

QMAKE_CXXFLAGS += /DWNT

README_FILE = PluginMeshSTEPLoader.txt

unix : QMAKE_POST_LINK = cp $$README_FILE $$DESTDIR 
win32 : QMAKE_POST_LINK = copy \"$$README_FILE\" \"$$SOFA_DIR/lib/sofa-plugins\"
