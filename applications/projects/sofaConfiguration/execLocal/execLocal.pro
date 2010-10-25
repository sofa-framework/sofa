SOFA_DIR=../../../..

TEMPLATE = app
TARGET = sofaLocalConfiguration

include($${SOFA_DIR}/sofa.cfg)

CONFIG += $$CONFIGPROJECTGUI 

contains(DEFINES, SOFA_QT4){
QT += qt3support 
}

QMAKE_LIBDIR *= ../lib

DESTDIR = $$SOFA_DIR/bin

win32{
LIBS += -lshell32
}

macx : {
        LIBS += -framework  CoreFoundation
	CONFIG +=app_bundle
	RC_FILE = sofaConfiguration.icns
	QMAKE_INFO_PLIST = Info.plist
        QMAKE_BUNDLE_DATA += $$APP_BUNDLE_DATA
}

# The following is a workaround to get KDevelop to detect the name of the program to start
unix {
	!macx: QMAKE_POST_LINK = ln -sf sofaLocalConfiguration$$SUFFIX $$DESTDIR/sofaLocalConfiguration-latest
}

# The following create enables to start Modeler from the command line as well as graphically
macx {
	QMAKE_POST_LINK = ln -sf sofaLocalConfiguration$$SUFFIX.app/Contents/MacOS/sofaLocalConfiguration$$SUFFIX $$DESTDIR/sofaLocalConfiguration$$SUFFIX
}

!macx : RC_FILE = sofa.rc

SOURCES = Main.cpp \
          ../lib/SofaConfiguration.cpp    \
          ../lib/ConfigurationParser.cpp          
HEADERS = ../lib/SofaConfiguration.h \
          ../lib/ConfigurationParser.h   
