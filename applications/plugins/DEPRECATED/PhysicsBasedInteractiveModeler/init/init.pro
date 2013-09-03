
######  GENERAL PLUGIN CONFIGURATION, you shouldn't have to modify it

SOFA_DIR=../../../..
TEMPLATE = lib

include($${SOFA_DIR}/sofa.cfg)

DESTDIR = $$SOFA_DIR/lib/sofa-plugins

#set configuration to dynamic library
contains (DEFINES, SOFA_QT4) {	

	  CONFIG += $$CONFIGLIBRARIES qt
	  !contains(CONFIGSTATIC, static) {
	CONFIG -= staticlib
          CONFIG += dll
}
	  QT += opengl qt3support xml
}
else {
	  CONFIG += $$CONFIGLIBRARIES qt
	  !contains(CONFIGSTATIC, static) {
	CONFIG -= staticlib
          CONFIG += dll
}
	  QT += opengl	
}

###### SPECIFIC PLUGIN CONFIGURATION, you should modify it to configure your plugin

TARGET = PhysicsBasedInteractiveModeler
DEFINES += SOFA_BUILD_PIM


LIBS += $$SOFA_FRAMEWORK_LIBS
LIBS += -lsofasimulation$$LIBSUFFIX
LIBS += -lsofacomponentlinearsolver$$LIBSUFFIX
LIBS += -lsofacomponentodesolver$$LIBSUFFIX
LIBS += -lsofacomponentmass$$LIBSUFFIX
LIBS += -lsofacomponentforcefield$$LIBSUFFIX
LIBS += -lsofacomponentcollision$$LIBSUFFIX
LIBS += -lsofacomponentconstraint$$LIBSUFFIX
LIBS += -lsofacomponentmisc$$LIBSUFFIX
LIBS += -lsofacomponentbase$$LIBSUFFIX
LIBS += -lsofacomponentmapping$$LIBSUFFIX
LIBS += -lsofacomponentengine$$LIBSUFFIX
LIBS += -lsofacomponentvisualmodel$$LIBSUFFIX
LIBS += -lpim$$LIBSUFFIX
LIBS += -lpimguiqt$$LIBSUFFIX
LIBS += -lsofaguiqt$$LIBSUFFIX

SOURCES = initPim.cpp

README_FILE = PhysicsBasedInteractiveModeler.txt

unix : QMAKE_POST_LINK = cp $$README_FILE $$DESTDIR 
win32 : QMAKE_POST_LINK = copy \"$$README_FILE\" \"$$SOFA_DIR/lib/sofa-plugins\"
