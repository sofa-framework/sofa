
######  GENERAL PLUGIN CONFIGURATION, you shouldn't have to modify it

SOFA_DIR=../../..
include($${SOFA_DIR}/sofa.cfg)
TEMPLATE = lib
DESTDIR = $$SOFA_DIR/bin



#set configuration to dynamic library
CONFIG += $$CONFIGLIBRARIES
CONFIG -= staticlib
CONFIG += dll

###### SPECIFIC PLUGIN CONFIGURATION, you should modify it to configure your plugin

TARGET = PluginExample$$LIBSUFFIX
DEFINES += SOFA_BUILD_PLUGINEXAMPLE

LIBS += $$SOFA_LIBS
LIBS += $$SOFA_EXT_LIBS
INCLUDEPATH += $$SOFA_DIR/extlibs

SOURCES = MyFakeComponent.cpp \
          OtherFakeComponent.cpp \
	      initPluginExample.cpp

HEADERS = MyFakeComponent.h \
          OtherFakeComponent.h 


#set a specific extension to easily recognize it as a sofa plugin
win32{	TARGET_EXT = .sll }
unix {
QMAKE_EXTENSION_SHLIB = sso
}
macx 
{	
QMAKE_EXTENSION_SHLIB = .sylib 
}







 


