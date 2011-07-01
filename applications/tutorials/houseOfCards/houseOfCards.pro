SOFA_DIR=../../..
TEMPLATE = app
TARGET = houseOfCards

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
QMAKE_LIBDIR += /usr/local/cuda/lib/
LIBS += -lsofaobjectcreator$$LIBSUFFIX
contains(CONFIGSTATIC, static) {
	LIBS += -Wl,--end-group
}


# The following is a workaround to get KDevelop to detect the name of the program to start
unix {
QMAKE_POST_LINK = ln -sf houseOfCards$$SUFFIX $$DESTDIR/houseOfCards-latest
}


SOURCES = Main.cpp 
HEADERS = 
