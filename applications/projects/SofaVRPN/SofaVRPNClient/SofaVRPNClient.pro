######  PLUGIN TARGET
TARGET = SofaVRPN
######  GENERAL PLUGIN CONFIGURATION, you shouldn't have to modify it

SOFA_DIR=../../../..
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

INCLUDEPATH += .
DEPENDPATH += .

LIBS += $$SOFA_GUI_LIBS
LIBS += $$SOFA_LIBS

LIBS += $$EXT_LIBS -lvrpn_client$$LIBSUFFIX

HEADERS +=  vrpnclient_config.h \
			VRPNDevice.h \
			VRPNAnalog.h \
			VRPNAnalog.inl \
			VRPNButton.h \
			VRPNTracker.h \
			VRPNTracker.inl \
			VRPNImager.h \
			VRPNImager.inl \
			WiimoteDriver.h \
			IRTracker.h \
			#ContactWarning.h \
			#ContactDisabler.h \
			ToolTracker.h \
			ToolTracker.inl 
   
SOURCES += 	initSofaVRPNClient.cpp \
			VRPNDevice.cpp \
			VRPNAnalog.cpp \
			VRPNButton.cpp \
		  	VRPNTracker.cpp \
		  	VRPNImager.cpp \
		  	WiimoteDriver.cpp \
		  	IRTracker.cpp \
		  	#ContactWarning.cpp \
 			#ContactDisabler.cpp \
		  	ToolTracker.cpp
