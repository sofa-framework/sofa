######  PLUGIN TARGET
TARGET = CGALPlugin

######  GENERAL PLUGIN CONFIGURATION, you shouldn't have to modify it

SOFA_DIR=../../..
TEMPLATE = lib

include($${SOFA_DIR}/sofa.cfg)

DESTDIR = $$SOFA_DIR/lib/sofa-plugins

#set configuration to dynamic library
CONFIG += $$CONFIGLIBRARIES
!contains(CONFIGSTATIC, static) {
	CONFIG -= staticlib
CONFIG += dll
}

#DEFINES += SOFA_NEW_CGAL_MESH

###### SPECIFIC PLUGIN CONFIGURATION, you should modify it to configure your plugin

DEFINES += SOFA_BUILD_CGALPLUGIN

LIBS += $$SOFA_LIBS
LIBS += $$SOFA_EXT_LIBS
INCLUDEPATH += $$SOFA_DIR/extlibs

SOURCES = initCGALPlugin.cpp \
		  MeshGenerationFromPolyhedron.cpp \
		  Optimize2DMesh.cpp \
		  Refine2DMesh.cpp \
                  RefineTriangleMesh.cpp \
		  TriangularConvexHull3D.cpp

HEADERS = \
		  MeshGenerationFromPolyhedron.h \
		  MeshGenerationFromPolyhedron.inl \
		  Optimize2DMesh.h \
		  Optimize2DMesh.inl \
		  Refine2DMesh.h \
		  Refine2DMesh.inl \
                  RefineTriangleMesh.h \
                  RefineTriangleMesh.inl \
		  TriangularConvexHull3D.h \
		  TriangularConvexHull3D.inl
		  
README_FILE = CGALPlugin.txt

unix{
        # These flags cause random crashes in CGAL mesher with gcc 4.4
	QMAKE_CFLAGS_RELEASE -= -fno-math-errno -funroll-loops -mfpmath=387
	QMAKE_CXXFLAGS_RELEASE -= -fno-math-errno -funroll-loops -mfpmath=387
}

unix : QMAKE_POST_LINK = cp $$README_FILE $$DESTDIR 
win32 : QMAKE_POST_LINK = copy \"$$README_FILE\" \"$$SOFA_DIR/lib/sofa-plugins\"


