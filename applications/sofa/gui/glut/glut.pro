# Target is a library: sofaguiglut
load(sofa/pre)

TEMPLATE = lib
TARGET = sofaguiglut

DEFINES += SOFA_BUILD_SOFAGUIGLUT
SOURCES = \
	SimpleGUI.cpp

HEADERS = \
	SimpleGUI.h

contains(DEFINES, SOFA_HAVE_BOOST) {
	SOURCES += MultithreadGUI.cpp
	HEADERS += MultithreadGUI.h
}

load(sofa/post)
