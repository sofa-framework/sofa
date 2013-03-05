load(sofa/pre)
defineAsPlugin(SensableEmulation)
######  GENERAL PLUGIN CONFIGURATION, you shouldnt have to modify it

TEMPLATE = lib

###### SPECIFIC PLUGIN CONFIGURATION, you should modify it to configure your plugin

TARGET = SensableEmulationPlugin
DEFINES += SOFA_BUILD_SENSABLEEMULATIONPLUGIN

SOURCES = \
#NewOmniDriverEmu.cpp \
initSensableEmulation.cpp \
OmniDriverEmu.cpp

HEADERS = \
#NewOmniDriverEmu.h \
OmniDriverEmu.h

README_FILE = PluginSensableEmulation.txt

#win32 {
#        LIBS *=
#}

#unix {
#        !macx : LIBS *= -lHD -lHL -lHDU -lHLU
#}


unix : QMAKE_POST_LINK = cp $$SRC_DIR/$$README_FILE $$LIB_DESTDIR 
win32 : QMAKE_POST_LINK = copy \"$$toWindowsPath($$SRC_DIR/$$README_FILE)\" \"$$LIB_DESTDIR\"

load(sofa/post)
