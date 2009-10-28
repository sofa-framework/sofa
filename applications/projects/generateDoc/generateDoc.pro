SOFA_DIR=../../..
TEMPLATE = app
TARGET = generateDoc

include($${SOFA_DIR}/sofa.cfg)

DESTDIR = $$SOFA_DIR/bin
CONFIG += $$CONFIGPROJECTCMD
#LIBS += $$SOFA_GUI_LIBS
LIBS += $$SOFA_LIBS

SOURCES = generateDoc.cpp \
          Main.cpp

HEADERS = generateDoc.h
