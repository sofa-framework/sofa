load(sofa/pre)
defineAsPlugin(CGALPlugin)

TARGET = CGALPlugin

#DEFINES += SOFA_NEW_CGAL_MESH

DEFINES += SOFA_BUILD_CGALPLUGIN

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

#TODO: add an install target for README files

unix : QMAKE_POST_LINK = cp $$SRC_DIR/$$README_FILE $$LIB_DESTDIR 
win32 : QMAKE_POST_LINK = copy \"$$toWindowsPath($$SRC_DIR/$$README_FILE)\" \"$$LIB_DESTDIR\"

unix{
        # These flags cause random crashes in CGAL mesher with gcc 4.4
	QMAKE_CFLAGS_RELEASE -= -fno-math-errno -funroll-loops -mfpmath=387
	QMAKE_CXXFLAGS_RELEASE -= -fno-math-errno -funroll-loops -mfpmath=387

        # CGAL generates many very very long warning messages...
	QMAKE_CFLAGS_RELEASE -= -Wall
	QMAKE_CXXFLAGS_RELEASE -= -Wall

        QMAKE_CXXFLAGS += -Wno-unused-parameter -fno-strict-aliasing
}

load(sofa/post)
