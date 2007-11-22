
SOFA_DIR=../../..
TEMPLATE = app

include($${SOFA_DIR}/sofa.cfg)

TARGET = generateDoc$$SUFFIX
DESTDIR = $$SOFA_DIR/bin
CONFIG += $$CONFIGPROJECTCMD
#LIBS += $$SOFA_GUI_LIBS
LIBS += $$SOFA_LIBS

SOURCES = generateDoc.cpp \
          Main.cpp

HEADERS = generateDoc.h
