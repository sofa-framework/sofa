SOFA_DIR=../../..
TEMPLATE = app
TARGET = sofaVerification

include($${SOFA_DIR}/sofa.cfg)

DESTDIR = $$SOFA_DIR/bin
CONFIG += $$CONFIGPROJECTCMD
#LIBS += $$SOFA_GUI_LIBS
LIBS += -Wl,--start-group
LIBS += $$SOFA_LIBS
LIBS += -Wl,--end-group

SOURCES = sofaVerification.cpp

HEADERS = 
