
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

TARGET = Xitact$$LIBSUFFIX
DEFINES += SOFA_BUILD_XITACTPLUGIN

LIBS += $$SOFA_LIBS
LIBS += $$SOFA_EXT_LIBS
LIBS += -lXiRobot
INCLUDEPATH += $$SOFA_DIR/extlibs/Xitact
INCLUDEPATH += $$SOFA_DIR/include

HEADERS = \
initXitact.h \
PaceMaker.h \
IHPDriver.h \
ITPDriver.h 


SOURCES = \
initXitact.cpp \
PaceMaker.cpp  \
IHPDriver.cpp \
ITPDriver.cpp 






README_FILE = PluginXitact.txt

unix : QMAKE_POST_LINK = cp $$README_FILE $$DESTDIR 
win32 : QMAKE_POST_LINK = copy \"$$README_FILE\" \"$$SOFA_DIR/lib/sofa-plugins\"
