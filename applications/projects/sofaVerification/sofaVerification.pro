SOFA_DIR=../../..
TEMPLATE = app
TARGET = sofaVerification

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

unix {
        LIBS += -ldl
}

SOURCES = sofaVerification.cpp

HEADERS = 
