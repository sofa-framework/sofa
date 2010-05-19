SOFA_DIR=../../..
TEMPLATE = app
TARGET = oneTetrahedron

include($${SOFA_DIR}/sofa.cfg)

DESTDIR = $$SOFA_DIR/bin
CONFIG += $$CONFIGPROJECTGUI
contains(CONFIGSTATIC, static) {
	LIBS += -Wl,--start-group
}
LIBS += $$SOFA_GUI_LIBS
contains(CONFIGSTATIC, static) {
	LIBS += -Wl,--end-group
	LIBS += -Wl,--start-group
}
LIBS += $$SOFA_LIBS
contains(CONFIGSTATIC, static) {
	LIBS += -Wl,--end-group
}

# The following is a workaround to get KDevelop to detect the name of the program to start
unix {
QMAKE_POST_LINK = ln -sf oneTetrahedron$$SUFFIX $$DESTDIR/oneTetrahedron-latest
}


SOURCES = Main.cpp 
HEADERS = 
