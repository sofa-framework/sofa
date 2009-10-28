SOFA_DIR=../../..
TEMPLATE = app
TARGET = SofaFlowVR

include($${SOFA_DIR}/sofa.cfg)

DESTDIR = $$SOFA_DIR/bin
CONFIG += $$CONFIGPROJECTGUI
LIBS += $$SOFA_GUI_LIBS
LIBS += $$SOFA_LIBS

SOURCES = Main.cpp 
HEADERS = 

########################################################################
#  FLOWVR
########################################################################
FLOWVR = $(FLOWVR_PREFIX)

QMAKE_CXXFLAGS += `pkg-config --cflags flowvr-mod flowvr-ftl flowvr-render`
QMAKE_LDFLAGS += `pkg-config --libs flowvr-mod flowvr-ftl flowvr-render`
QMAKE_LFLAGS_DEBUG+= `pkg-config --libs flowvr-mod flowvr-ftl flowvr-render`
QMAKE_LFLAGS_RELEASE+= `pkg-config --libs flowvr-mod flowvr-ftl flowvr-render`
#contains (CONFIGDEBUG, debug) {
#    DEFINES += FLOWVR_DEBUG DEBUG VERBOSE_ENABLE
#}

########################################################################
