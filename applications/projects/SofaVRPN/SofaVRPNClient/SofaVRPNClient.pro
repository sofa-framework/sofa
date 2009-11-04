######  PLUGIN TARGET
TARGET = SofaVRPN
######  GENERAL PLUGIN CONFIGURATION, you shouldn't have to modify it

SOFA_DIR=../../../..
TEMPLATE = lib

include($${SOFA_DIR}/sofa.cfg)

DESTDIR = $$SOFA_DIR/lib/sofa-plugins

#set configuration to dynamic library
CONFIG += $$CONFIGLIBRARIES
CONFIG -= staticlib
CONFIG += dll


###### SPECIFIC PLUGIN CONFIGURATION, you should modify it to configure your plugin

DEPENDPATH += .
INCLUDEPATH += .

LIBS += $$SOFA_GUI_LIBS
LIBS += $$SOFA_LIBS

LIBS += $$EXT_LIBS

HEADERS +=  vrpnclient_config.h \
			VRPNDevice.h \
			VRPNAnalog.h \
			VRPNButton.h \
			VRPNTracker.h
   
SOURCES += 	initSofaVRPNClient.cpp \
			VRPNDevice.cpp \
			VRPNAnalog.cpp \
			VRPNButton.cpp \
		  	VRPNTracker.cpp
