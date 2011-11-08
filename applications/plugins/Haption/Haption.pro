load(sofa/pre)
defineAsPlugin(Haption)
######  GENERAL PLUGIN CONFIGURATION, you shouldn't have to modify it

TEMPLATE = lib


###### SPECIFIC PLUGIN CONFIGURATION, you should modify it to configure your plugin

TARGET = HaptionPlugin
DEFINES += SOFA_BUILD_HAPTIONPLUGIN

win32 {
#	CONFIG(debug, debug|release) {
		QMAKE_LFLAGS += /NODEFAULTLIB:libcmtd
#	} else {
		QMAKE_LFLAGS += /NODEFAULTLIB:libcmt
#	}
}

SOURCES = \
initHaption.cpp \
HaptionDriver.cpp

HEADERS = \
HaptionDriver.h

README_FILE = PluginHaption.txt

unix : QMAKE_POST_LINK = cp $$SRC_DIR/$$README_FILE $$LIB_DESTDIR 
win32 : QMAKE_POST_LINK = copy \"$$toWindowsPath($$SRC_DIR/$$README_FILE)\" \"$$LIB_DESTDIR\"

load(sofa/post)
