SOFA_DIR=../../..
TEMPLATE = app
TARGET = generateDoc

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

SOURCES = generateDoc.cpp \
          Main.cpp

HEADERS = generateDoc.h
