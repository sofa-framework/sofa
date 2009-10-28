SOFA_DIR=../../..
TEMPLATE = app
TARGET = sofaBatch

include($${SOFA_DIR}/sofa.cfg)

DESTDIR = $$SOFA_DIR/bin
CONFIG += $$CONFIGPROJECTCMD
LIBS += $$SOFA_LIBS

SOURCES = sofaBatch.cpp

HEADERS = 
