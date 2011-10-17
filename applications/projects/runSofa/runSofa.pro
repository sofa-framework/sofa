load(sofa/pre)

TEMPLATE = app
TARGET = runSofa

CONFIG += console

macx {
	CONFIG += app_bundle
	RC_FILE = runSOFA.icns
	QMAKE_INFO_PLIST = Info.plist
    QMAKE_BUNDLE_DATA += $$APP_BUNDLE_DATA
} else {
	RC_FILE = sofa.rc
}

unix {
   LIBS += -ldl
}

SOURCES = Main.cpp 
HEADERS = 

load(sofa/post)
