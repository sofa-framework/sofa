SOFA_DIR=../../..
TEMPLATE = app
TARGET = sofaInitTimer

include($${SOFA_DIR}/sofa.cfg)

DESTDIR = $$SOFA_DIR/bin
CONFIG += $$CONFIGPROJECTCMD
contains(CONFIGSTATIC, static) {
	LIBS += -Wl,--start-group
}
LIBS += $$SOFA_GUI_LIBS
contains(CONFIGSTATIC, static) {
	LIBS += -Wl,--end-group
	LIBS += -Wl,--start-group
	LIBS += -Wl,--whole-archive
}
LIBS += $$SOFA_LIBS
contains(CONFIGSTATIC, static) {
	LIBS += -Wl,--no-whole-archive
	LIBS += -Wl,--end-group
}

unix {
        LIBS += -ldl
}

SOURCES = sofaInitTimer.cpp

HEADERS = 
