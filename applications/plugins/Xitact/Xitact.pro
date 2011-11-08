
######  GENERAL PLUGIN CONFIGURATION, you shouldn't have to modify it

load(sofa/pre)
defineAsPlugin(Xitact)
TEMPLATE = lib

###### SPECIFIC PLUGIN CONFIGURATION, you should modify it to configure your plugin

TARGET = XitactPlugin
DEFINES += SOFA_BUILD_XITACTPLUGIN

LIBS += -lXiRobot

HEADERS = \
XiTrocarInterface.h \
initXitact.h \
PaceMaker.h \
IHPDriver.h \
ITPDriver.h 


SOURCES = \
initXitact.cpp \
PaceMaker.cpp  \
IHPDriver.cpp \
ITPDriver.cpp 


README_FILE = PluginXitact.txt

unix : QMAKE_POST_LINK = cp $$SRC_DIR/$$README_FILE $$LIB_DESTDIR 
win32 : QMAKE_POST_LINK = copy \"$$toWindowsPath($$SRC_DIR/$$README_FILE)\" \"$$LIB_DESTDIR\"

load(sofa/post)
