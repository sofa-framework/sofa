
######  GENERAL PLUGIN CONFIGURATION, you shouldn't have to modify it
load(sofa/pre)
defineAsPlugin(Sensable)


TEMPLATE = lib
TARGET = SensablePlugin

DEFINES += SOFA_BUILD_SENSABLEPLUGIN


#set configuration to dynamic library
contains (DEFINES, SOFA_QT4) {	
	CONFIG += qt 
	QT += opengl qt3support xml
}
else {
	CONFIG += qt
	QT += opengl
}





SOURCES = \
initSensable.cpp \
NewOmniDriver.cpp

HEADERS = \
NewOmniDriver.h

README_FILE = PluginSensable.txt
unix : QMAKE_POST_LINK = cp $$SRC_DIR/$$README_FILE $$LIB_DESTDIR 
win32 : QMAKE_POST_LINK = copy \"$$toWindowsPath($$SRC_DIR/$$README_FILE)\" \"$$LIB_DESTDIR"

load(sofa/post)
