######  PLUGIN TARGET
TARGET = Voxelizer

######  GENERAL PLUGIN CONFIGURATION, you shouldn't have to modify it

SOFA_DIR=../../..
TEMPLATE = lib

include($${SOFA_DIR}/sofa.cfg)

DESTDIR = $$SOFA_DIR/lib/sofa-plugins

#set configuration to dynamic library
CONFIG += $$CONFIGLIBRARIES
!contains(CONFIGSTATIC, static) {
	CONFIG -= staticlib
CONFIG += dll
}

###### SPECIFIC PLUGIN CONFIGURATION, you should modify it to configure your plugin

DEFINES += SOFA_BUILD_VOXELIZER

LIBS += $$SOFA_LIBS
LIBS += $$SOFA_EXT_LIBS
INCLUDEPATH += $$SOFA_DIR/extlibs
DEPENDPATH += $$SOFA_DIR/extlibs

 LIBS += -lsofagpucuda$$LIBSUFFIX

 HEADERS += initVoxelizer.h \
            Voxelizer.h \
            Voxelizer.inl 
 SOURCES += initVoxelizer.cpp \
            Voxelizer.cpp 

README_FILE = Voxelizer.txt

unix : QMAKE_POST_LINK = cp $$README_FILE $$DESTDIR 
win32 : QMAKE_POST_LINK = copy \"$$README_FILE\" \"$$SOFA_DIR/lib/sofa-plugins\"


