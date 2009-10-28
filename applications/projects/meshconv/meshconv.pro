SOFA_DIR=../../..
TEMPLATE = app
TARGET = meshconv

include($${SOFA_DIR}/sofa.cfg)

DESTDIR = $$SOFA_DIR/bin
CONFIG += $$CONFIGPROJECTCMD
#LIBS += $$SOFA_GUI_LIBS
LIBS += $$SOFA_LIBS

SOURCES = meshconv.cpp tesselate.cpp

HEADERS = 
