SOFA_DIR=../../..
TEMPLATE = app
TARGET = sofaVerification

include($${SOFA_DIR}/sofa.cfg)

DESTDIR = $$SOFA_DIR/bin
CONFIG += $$CONFIGPROJECTCMD
#LIBS += $$SOFA_GUI_LIBS
LIBS += $$SOFA_LIBS

SOURCES = sofaVerification.cpp

HEADERS = 
