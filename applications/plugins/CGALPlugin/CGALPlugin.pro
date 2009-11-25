######  PLUGIN TARGET
TARGET = CGALPlugin

######  GENERAL PLUGIN CONFIGURATION, you shouldn't have to modify it

SOFA_DIR=../../..
TEMPLATE = lib

include($${SOFA_DIR}/sofa.cfg)

DESTDIR = $$SOFA_DIR/lib/sofa-plugins

#set configuration to dynamic library
CONFIG += $$CONFIGLIBRARIES
CONFIG -= staticlib
CONFIG += dll

DEFINES += SOFA_NEW_CGAL_MESH

###### SPECIFIC PLUGIN CONFIGURATION, you should modify it to configure your plugin

DEFINES += SOFA_BUILD_CGALPLUGIN

LIBS += $$SOFA_LIBS
LIBS += $$SOFA_EXT_LIBS
INCLUDEPATH += $$SOFA_DIR/extlibs

SOURCES = initCGALPlugin.cpp \
		  MeshGenerationFromPolyhedron.cpp \
		  TriangularConvexHull3D.cpp

HEADERS = \
		  MeshGenerationFromPolyhedron.h \
		  TriangularConvexHull3D.h
		  
README_FILE = CGALPlugin.txt

unix{
        # These flags cause random crashes in CGAL mesher with gcc 4.4
	QMAKE_CFLAGS_RELEASE -= -fno-math-errno -funroll-loops -mfpmath=387
	QMAKE_CXXFLAGS_RELEASE -= -fno-math-errno -funroll-loops -mfpmath=387
}

unix : QMAKE_POST_LINK = cp $$README_FILE $$DESTDIR 
win32 : QMAKE_POST_LINK = copy \"$$README_FILE\" \"$$SOFA_DIR/lib/sofa-plugins\"


