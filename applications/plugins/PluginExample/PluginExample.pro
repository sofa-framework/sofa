load(sofa/pre)
defineAsPlugin(PluginExample)

TARGET = PluginExample

#set configuration to dynamic library

contains (DEFINES, SOFA_QT4) {	
	CONFIG += qt 
	QT += opengl qt3support xml
}
else {
	CONFIG += qt
	QT += opengl
}

DEFINES += SOFA_BUILD_PLUGINEXAMPLE

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

#TODO: add an install target for README files

unix : QMAKE_POST_LINK = cp $$SRC_DIR/$$README_FILE $$LIB_DESTDIR 
win32 : QMAKE_POST_LINK = copy \"$$toWindowsPath($$SRC_DIR/$$README_FILE)\" \"$$LIB_DESTDIR"

load(sofa/post)
