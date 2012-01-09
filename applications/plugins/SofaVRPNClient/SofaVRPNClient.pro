load(sofa/pre)
defineAsPlugin(SofaVRPNClient)
######  PLUGIN TARGET
TARGET = SofaVRPNClient

DEFINES += SOFA_BUILD_SOFAVRPNCLIENT

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

load(sofa/post)
