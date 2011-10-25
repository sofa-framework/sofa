load(sofa/pre)

TEMPLATE = app
TARGET = Modeler

CONFIG += console

INCLUDEPATH += $$BUILD_DIR/../lib/$$UI_DIR # HACK: some uic generated headers are generated in another .pro
DEPENDPATH += $$BUILD_DIR/../lib/$$UI_DIR # HACK: some uic generated headers are generated in another .pro

macx : {
	CONFIG +=app_bundle
	RC_FILE = Modeler.icns
	QMAKE_INFO_PLIST = Info.plist
        QMAKE_BUNDLE_DATA += $$APP_BUNDLE_DATA
#	QMAKE_POST_LINK = cp -r ../../../../share/* ../../../../bin/Modeler$$SUFFIX.app/Contents/Resources/.
}

!macx : RC_FILE = sofa.rc

SOURCES = Main.cpp
HEADERS = 

load(sofa/post)
