SOFA_DIR=../../..
TEMPLATE = app

include($${SOFA_DIR}/sofa.cfg)

TARGET = runSofa$$SUFFIX
DESTDIR = $$SOFA_DIR/bin
CONFIG += $$CONFIGPROJECTGUI 
LIBS += $$SOFA_GUI_LIBS
LIBS += $$SOFA_LIBS

macx : {
	CONFIG +=app_bundle 
	RC_FILE = runSOFA.icns
}

# The following is a workaround to get KDevelop to detect the name of the program to start
unix {
	!macx: QMAKE_POST_LINK = ln -sf runSofa$$SUFFIX $$DESTDIR/runSofa-latest
}


!macx : RC_FILE = sofa.rc

SOURCES = Main.cpp 
HEADERS = 
