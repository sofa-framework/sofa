
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

TARGET = SensablePlugin$$LIBSUFFIX
DEFINES += SOFA_BUILD_SENSABLEPLUGIN

LIBS += $$SOFA_LIBS
LIBS += $$SOFA_EXT_LIBS
INCLUDEPATH += $$SOFA_DIR/extlibs/Sensable

SOURCES = \
initSensable.cpp \
NewOmniDriver.cpp

HEADERS = \
NewOmniDriver.h

README_FILE = PluginSensable.txt

unix : QMAKE_POST_LINK = cp $$README_FILE $$DESTDIR 
win32 : QMAKE_POST_LINK = copy \"$$README_FILE\" \"$$SOFA_DIR/lib/sofa-plugins\"
