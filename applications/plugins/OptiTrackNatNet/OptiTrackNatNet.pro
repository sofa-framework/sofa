load(sofa/pre)
defineAsPlugin(OptiTrackNatNet)
######  PLUGIN TARGET
TARGET = OptiTrackNatNet

DEFINES += SOFA_BUILD_OPTITRACKNATNET

HEADERS +=  initOptiTrackNatNet.h \
			OptiTrackNatNetClient.h
   
SOURCES += 	initOptiTrackNatNet.cpp \
			OptiTrackNatNetClient.cpp

load(sofa/post)
