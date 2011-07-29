load(sofa/pre)

TEMPLATE = app
TARGET = Modeler

INCLUDEPATH += $$BUILD_DIR/../lib/$$UI_DIR # HACK: some uic generated headers are generated in another .pro

macx : {
	CONFIG +=app_bundle
	RC_FILE = Modeler.icns
	QMAKE_INFO_PLIST = Info.plist
        QMAKE_BUNDLE_DATA += $$APP_BUNDLE_DATA
#	QMAKE_POST_LINK = cp -r ../../../../share/* ../../../../bin/Modeler$$SUFFIX.app/Contents/Resources/.
}

# The following is a workaround to get KDevelop to detect the name of the program to start
unix {
	!macx: QMAKE_POST_LINK = ln -sf Modeler$$SUFFIX $$APP_DESTDIR/Modeler-latest
}

# The following create enables to start Modeler from the command line as well as graphically
macx {
	QMAKE_POST_LINK = ln -sf Modeler$$SUFFIX.app/Contents/MacOS/Modeler$$SUFFIX $$APP_DESTDIR/Modeler$$SUFFIX
}

!macx : RC_FILE = sofa.rc

SOURCES = Main.cpp
HEADERS = 

load(sofa/post)
