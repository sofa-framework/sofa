SOFA_DIR=../../..
TEMPLATE = app
TARGET = SofaProjectTest

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
	LIBS += -Wl,--whole-archive
}
LIBS += $$SOFA_LIBS
contains(CONFIGSTATIC, static) {
	LIBS += -Wl,--no-whole-archive
	LIBS += -Wl,--end-group
}

macx : {
	CONFIG +=app_bundle
	RC_FILE = runSOFA.icns
	QMAKE_INFO_PLIST = Info.plist
        QMAKE_BUNDLE_DATA += $$APP_BUNDLE_DATA
#	QMAKE_POST_LINK = cp -r ../../../share/* ../../../bin/runSofa$$SUFFIX.app/Contents/Resources/.
}

unix {
        # The following is a workaround to get KDevelop to detect the name of the program to start
	!macx: QMAKE_POST_LINK = ln -sf runSofa$$SUFFIX $$DESTDIR/runSofa-latest
}

macx {
        # The following create enables to start the program from the command line as well as graphically
	QMAKE_POST_LINK = ln -sf "$$TARGET".app/Contents/MacOS/"$$TARGET" $$DESTDIR/"$$TARGET" ;
}

# !macx : RC_FILE = sofa.rc

SOURCES = Main.cpp 
HEADERS = 
