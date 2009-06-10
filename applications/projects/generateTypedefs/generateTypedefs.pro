SOFA_DIR=../../..
TEMPLATE = app

include($${SOFA_DIR}/sofa.cfg)

TARGET = generateTypedefs$$SUFFIX
DESTDIR = $$SOFA_DIR/bin
CONFIG += $$CONFIGPROJECTCMD
LIBS += $$SOFA_LIBS

SOURCES = Main.cpp

HEADERS = 
