SOFA_DIR=../../..
TEMPLATE = app
TARGET = chainHybrid

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
LIBS += -lsofaobjectcreator$$LIBSUFFIX
contains(CONFIGSTATIC, static) {
	LIBS += -Wl,--end-group
}

contains( DEFINES,SOFA_HAS_BOOST_KERNEL) { 
LIBS += -lsofaBoostKernel$$LIBSUFFIX
}

# The following is a workaround to get KDevelop to detect the name of the program to start
unix {
QMAKE_POST_LINK = ln -sf chainHybrid$$SUFFIX $$DESTDIR/chainHybrid-latest
}


SOURCES = Main.cpp 
HEADERS = 
