######  PLUGIN TARGET
TARGET = pim

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

TARGET = pim
DEFINES += SOFA_BUILD_PIM

LIBS += $$SOFA_FRAMEWORK_LIBS
LIBS += $$SOFA_LIBS
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
LIBS += -lCGALPlugin$$LIBSUFFIX
#LIBS += -lTriangularMeshRefiner$$LIBSUFFIX
LIBS += -lsofagui$$LIBSUFFIX
LIBS += -lsofaguimain$$LIBSUFFIX
LIBS += -lsofaguimain$$LIBSUFFIX

SOURCES = ProgressiveScaling.cpp \
          ComputeMeshIntersection.cpp \
          Parameters.cpp \
          TransformPlaneConstraint.cpp \
          EventManager.cpp \
          SculptBodyPerformer.cpp

HEADERS = SculptBodyPerformer.h \
          SculptBodyPerformer.inl \
          ProgressiveScaling.h \
          ProgressiveScaling.inl \
          ComputeMeshIntersection.h \
          ComputeMeshIntersection.inl \
          Parameters.h \
          TransformPlaneConstraint.h \
          EventManager.inl \
          EventManager.h
