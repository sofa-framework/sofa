SOFA_DIR=../../..
TEMPLATE = app

include($$SOFA_DIR/sofa.cfg)

TARGET = runSofa$$SUFFIX
DESTDIR = $$SOFA_DIR/bin
CONFIG += $$CONFIGPROJECTGUI
LIBS += $$SOFA_GUI_LIBS
LIBS += $$SOFA_LIBS

# The following is a workaround to get KDevelop to detect the name of the program to start
unix {
	QMAKE_POST_LINK = ln -sf runSofa$$SUFFIX $$DESTDIR/runSofa-latest
}


RC_FILE = sofa.rc

SOURCES = Main.cpp 
HEADERS = 
