load(sofa/pre)

TEMPLATE = app
TARGET = runSofa

CONFIG += console

macx {
	CONFIG += app_bundle
	RC_FILE = runSOFA.icns
	QMAKE_INFO_PLIST = Info.plist
  QMAKE_BUNDLE_DATA += $$APP_BUNDLE_DATA
  
  # The following create enables to start the program from the command line as well as graphically
  QMAKE_POST_LINK = ln -sf $${TARGET}.$${TEMPLATE}/Contents/MacOS/$$TARGET $$APP_DESTDIR/$$TARGET ;
} else {
	RC_FILE = sofa.rc
}

unix {
   LIBS += -ldl
   # The following is a workaround to get KDevelop to detect the name of the program to start
  !macx: QMAKE_POST_LINK = ln -sf runSofa$$APPSUFFIX $$APP_DESTDIR/runSofa-latest
}

SOURCES = Main.cpp 
HEADERS = 

load(sofa/post)
