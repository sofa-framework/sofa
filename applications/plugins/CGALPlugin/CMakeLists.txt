cmake_minimum_required(VERSION 2.8)

project("SofaCGALPlugin")

include(${SOFA_CMAKE_DIR}/preProject.cmake)


set(HEADER_FILES

    initCGALPlugin.h 
	MeshGenerationFromPolyhedron.h 
    MeshGenerationFromPolyhedron.inl 
    TriangularConvexHull3D.h 
    TriangularConvexHull3D.inl 
    DecimateMesh.h 
    DecimateMesh.inl

	)

set(SOURCE_FILES

    initCGALPlugin.cpp 
    MeshGenerationFromPolyhedron.cpp 
    TriangularConvexHull3D.cpp 
    DecimateMesh.cpp

	)

set(README_FILE  CGALPlugin.txt)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_CGALPLUGIN")
set(LINKER_DEPENDENCIES SofaGuiQt cgal boost )

include(${SOFA_CMAKE_DIR}/postProject.cmake)