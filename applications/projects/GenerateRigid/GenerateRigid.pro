SOFA_DIR=../../..
TEMPLATE = app
TARGET = GenerateRigid

include($${SOFA_DIR}/sofa.cfg)

DESTDIR = $$SOFA_DIR/bin
CONFIG += $$CONFIGPROJECTCMD
#LIBS += $$SOFA_GUI_LIBS
contains(CONFIGSTATIC, static) {
LIBS += -Wl,--start-group
}
LIBS += $$SOFA_LIBS
contains(CONFIGSTATIC, static) {
LIBS += -Wl,--end-group
}

SOURCES = GenerateRigid.cpp \
          Main.cpp

HEADERS = GenerateRigid.h
