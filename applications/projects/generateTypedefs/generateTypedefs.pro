SOFA_DIR=../../..
TEMPLATE = app
TARGET = generateTypedefs

include($${SOFA_DIR}/sofa.cfg)

DESTDIR = $$SOFA_DIR/bin
CONFIG += $$CONFIGPROJECTCMD
contains(CONFIGSTATIC, static) {
LIBS += -Wl,--start-group
LIBS += $$SOFA_GUI_LIBS
LIBS += -Wl,--end-group
LIBS += -Wl,--start-group
LIBS += $$SOFA_LIBS
LIBS += -Wl,--end-group
}
else {
LIBS += $$SOFA_GUI_LIBS
LIBS += $$SOFA_LIBS
}

SOURCES = Main.cpp

HEADERS = 
