SOFA_DIR=../../..
TEMPLATE = app

include($$SOFA_DIR/sofa.cfg)

TARGET = GenerateRigid$$SUFFIX
DESTDIR = $$SOFA_DIR/bin
CONFIG += $$CONFIGPROJECTCMD
LIBS += $$SOFA_GUI_LIBS
LIBS += $$SOFA_LIBS

SOURCES = GenerateRigid.cpp \
          Main.cpp

HEADERS = GenerateRigid.h
