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
		  Refine2DMesh.cpp \
                  RefineTriangleMesh.cpp \
		  TriangularConvexHull3D.cpp \
          DecimateMesh.cpp

HEADERS = \
		  MeshGenerationFromPolyhedron.h \
		  MeshGenerationFromPolyhedron.inl \
		  Refine2DMesh.h \
		  Refine2DMesh.inl \
                  RefineTriangleMesh.h \
                  RefineTriangleMesh.inl \
		  TriangularConvexHull3D.h \
		  TriangularConvexHull3D.inl \
          DecimateMesh.h \
          DecimateMesh.inl

# These files do not compile with current version of CGAL...
contains(DEFINES, SOFA_NEW_CGAL_MESH) {
  SOURCES += Optimize2DMesh.cpp
  
  HEADERS += Optimize2DMesh.h \
             Optimize2DMesh.inl
}

README_FILE = CGALPlugin.txt

unix{
        # These flags cause random crashes in CGAL mesher with gcc 4.4
	QMAKE_CFLAGS_RELEASE -= -fno-math-errno -funroll-loops -mfpmath=387
	QMAKE_CXXFLAGS_RELEASE -= -fno-math-errno -funroll-loops -mfpmath=387

        # CGAL generates many very very long warning messages...
	QMAKE_CFLAGS_RELEASE -= -Wall
	QMAKE_CXXFLAGS_RELEASE -= -Wall

        QMAKE_CXXFLAGS += -Wno-unused-parameter -Wno-array-bounds -fno-strict-aliasing
}

unix : QMAKE_POST_LINK = cp $$README_FILE $$DESTDIR 
win32 : QMAKE_POST_LINK = copy \"$$README_FILE\" \"$$SOFA_DIR/lib/sofa-plugins\"


