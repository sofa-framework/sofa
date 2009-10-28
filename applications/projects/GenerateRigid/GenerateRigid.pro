SOFA_DIR=../../..
TEMPLATE = app
TARGET = GenerateRigid

include($${SOFA_DIR}/sofa.cfg)

DESTDIR = $$SOFA_DIR/bin
CONFIG += $$CONFIGPROJECTCMD
#LIBS += $$SOFA_GUI_LIBS
LIBS += $$SOFA_LIBS

SOURCES = GenerateRigid.cpp \
          Main.cpp

HEADERS = GenerateRigid.h
