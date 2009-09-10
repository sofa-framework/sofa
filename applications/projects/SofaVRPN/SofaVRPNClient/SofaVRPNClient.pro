######  GENERAL PLUGIN CONFIGURATION, you shouldn't have to modify it

SOFA_DIR=../../../..
TEMPLATE = lib
DESTDIR = $$SOFA_DIR/bin

include($${SOFA_DIR}/sofa.cfg)

#set configuration to dynamic library
CONFIG += $$CONFIGLIBRARIES
CONFIG -= staticlib
CONFIG += dll

#set a specific extension to easily recognize it as a sofa plugin
win32{	TARGET_EXT = .sll }
unix {	QMAKE_EXTENSION_SHLIB = sso }
macx {	QMAKE_EXTENSION_SHLIB = sylib }



###### SPECIFIC PLUGIN CONFIGURATION, you should modify it to configure your plugin
TARGET = SofaVRPN

DEPENDPATH += .
INCLUDEPATH += .

LIBS += $$SOFA_GUI_LIBS
LIBS += $$SOFA_LIBS

LIBS += $$EXT_LIBS

HEADERS +=  VRPNDevice.h \
			VRPNTracker.h
   
SOURCES += 	initSofaVRPNClient.cpp \
			VRPNDevice.cpp \
		  	VRPNTracker.cpp
