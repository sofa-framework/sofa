load(sofa/pre)
defineAsPlugin(OptiTrackNatNet)
######  PLUGIN TARGET
TARGET = OptiTrackNatNet

DEFINES += SOFA_BUILD_OPTITRACKNATNET

HEADERS +=  initOptiTrackNatNet.h \
			OptiTrackNatNetClient.h \
			OptiTrackNatNetDevice.h
   
SOURCES += 	initOptiTrackNatNet.cpp \
			OptiTrackNatNetClient.cpp \
			OptiTrackNatNetDevice.cpp

load(sofa/post)
