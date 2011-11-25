load(sofa/pre)
defineAsPlugin(Sensable)
######  GENERAL PLUGIN CONFIGURATION, you shouldn't have to modify it

TEMPLATE = lib

###### SPECIFIC PLUGIN CONFIGURATION, you should modify it to configure your plugin

TARGET = SensablePlugin
DEFINES += SOFA_BUILD_SENSABLEPLUGIN

SOURCES = \
initSensable.cpp \
NewOmniDriver.cpp

HEADERS = \
NewOmniDriver.h

README_FILE = PluginSensable.txt

#win32 {
#        LIBS *=
#}

unix {
        !macx : LIBS *= -lHD -lHL -lHDU -lHLU
}


unix : QMAKE_POST_LINK = cp $$SRC_DIR/$$README_FILE $$LIB_DESTDIR 
win32 : QMAKE_POST_LINK = copy \"$$toWindowsPath($$SRC_DIR/$$README_FILE)\" \"$$LIB_DESTDIR\"

load(sofa/post)
