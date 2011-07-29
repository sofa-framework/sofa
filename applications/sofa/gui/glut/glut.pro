# Target is a library: sofaguiglut
load(sofa/pre)

TEMPLATE = lib
TARGET = sofaguiglut

DEFINES += SOFA_BUILD_SOFAGUIGLUT

SOURCES = \
	    SimpleGUI.cpp

HEADERS = \
	    SimpleGUI.h

load(sofa/post)
