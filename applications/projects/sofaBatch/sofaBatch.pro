SOFA_DIR=../../..
TEMPLATE = app

include($${SOFA_DIR}/sofa.cfg)

TARGET = sofaBatch$$SUFFIX
DESTDIR = $$SOFA_DIR/bin
CONFIG += $$CONFIGPROJECTCMD
LIBS += $$SOFA_LIBS

SOURCES = sofaBatch.cpp

HEADERS = 
