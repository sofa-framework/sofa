SOFA_DIR=../../../..
include($${SOFA_DIR}/sofa.cfg)

CONFIG += $$CONFIGPROJECTGUI 
contains(DEFINES, SOFA_QT4){
QT += qt3support 
}

TEMPLATE = app

QMAKE_LIBDIR = ../lib

TARGET = sofaConfiguration
DESTDIR = $$SOFA_DIR/bin


macx : {
	CONFIG +=app_bundle
	RC_FILE = sofaConfiguration.icns
	QMAKE_INFO_PLIST = Info.plist
        QMAKE_BUNDLE_DATA += $$APP_BUNDLE_DATA
}

# The following is a workaround to get KDevelop to detect the name of the program to start
unix {
	!macx: QMAKE_POST_LINK = ln -sf sofaConfiguration$$SUFFIX $$DESTDIR/sofaConfiguration-latest
}

# The following create enables to start Modeler from the command line as well as graphically
macx {
	QMAKE_POST_LINK = ln -sf sofaConfiguration.app/Contents/MacOS/sofaConfiguration$$SUFFIX $$DESTDIR/sofaConfiguration$$SUFFIX
}

!macx : RC_FILE = sofa.rc

SOURCES = Main.cpp \
          SofaConfiguration.cpp           
HEADERS = SofaConfiguration.h
