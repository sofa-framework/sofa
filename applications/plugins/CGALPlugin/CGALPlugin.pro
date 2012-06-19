load(sofa/pre)
defineAsPlugin(CGALPlugin)

TARGET = CGALPlugin

DEFINES += SOFA_BUILD_CGALPLUGIN

SOURCES = initCGALPlugin.cpp \
		  MeshGenerationFromPolyhedron.cpp \
		  TriangularConvexHull3D.cpp \
          DecimateMesh.cpp

HEADERS = initCGALPlugin.h \
		  MeshGenerationFromPolyhedron.h \
		  MeshGenerationFromPolyhedron.inl \
		  TriangularConvexHull3D.h \
		  TriangularConvexHull3D.inl \
          DecimateMesh.h \
          DecimateMesh.inl

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
