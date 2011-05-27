
######  GENERAL PLUGIN CONFIGURATION, you shouldnt have to modify it

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

TARGET = SensablePluginEmulation$$LIBSUFFIX
DEFINES += SOFA_BUILD_SENSABLEPLUGIN

LIBS += $$SOFA_LIBS
LIBS += $$SOFA_EXT_LIBS

SOURCES = \
initSensableEmulation.cpp \
NewOmniDriverEmu.cpp

HEADERS = \
NewOmniDriverEmu.h

README_FILE = PluginSensableEmulation.txt

unix : QMAKE_POST_LINK = cp $$README_FILE $$DESTDIR 
win32 : QMAKE_POST_LINK = copy \"$$README_FILE\" \"$$SOFA_DIR/lib/sofa-plugins\"
