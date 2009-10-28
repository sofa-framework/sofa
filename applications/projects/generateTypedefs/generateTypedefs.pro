SOFA_DIR=../../..
TEMPLATE = app
TARGET = generateTypedefs

include($${SOFA_DIR}/sofa.cfg)

DESTDIR = $$SOFA_DIR/bin
CONFIG += $$CONFIGPROJECTCMD
LIBS += $$SOFA_LIBS

SOURCES = Main.cpp

HEADERS = 
