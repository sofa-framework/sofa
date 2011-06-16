######  PLUGIN TARGET
TARGET = PluginExample

######  GENERAL PLUGIN CONFIGURATION, you shouldn't have to modify it

SOFA_DIR=../../..
TEMPLATE = lib

include($${SOFA_DIR}/sofa.cfg)

DESTDIR = $$SOFA_DIR/lib/sofa-plugins

#set configuration to dynamic library

contains (DEFINES, SOFA_QT4) {	
	CONFIG += $$CONFIGLIBRARIES qt 
	QT += opengl qt3support xml
}
else{
	CONFIG += $$CONFIGLIBRARIES qt
	QT += opengl
}
!contains(CONFIGSTATIC, static) {
	CONFIG -= staticlib
CONFIG += dll
}

###### SPECIFIC PLUGIN CONFIGURATION, you should modify it to configure your plugin

DEFINES += SOFA_BUILD_PLUGINEXAMPLE

LIBS += $$SOFA_LIBS
LIBS += $$SOFA_GUI_LIBS
LIBS += $$SOFA_EXT_LIBS

INCLUDEPATH += $$SOFA_DIR/extlibs

SOURCES = MyBehaviorModel.cpp \
          MyDataWidgetUnsigned.cpp \
          MyProjectiveConstraintSet.cpp \
          MyMappingPendulumInPlane.cpp \
          initPlugin.cpp

HEADERS = MyBehaviorModel.h \
          MyDataWidgetUnsigned.h \
          MyMappingPendulumInPlane.h \
          MyMappingPendulumInPlane.inl \
          MyProjectiveConstraintSet.h \
          MyProjectiveConstraintSet.inl \
		  initPlugin.h
		  
README_FILE = PluginExample.txt

unix : QMAKE_POST_LINK = cp $$README_FILE $$DESTDIR 
win32 : QMAKE_POST_LINK = copy \"$$README_FILE\" \"$$SOFA_DIR/lib/sofa-plugins\"


