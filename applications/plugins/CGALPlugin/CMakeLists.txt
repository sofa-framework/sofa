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

set(README_FILES

	CGALPlugin.txt
	)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

find_package(CGAL REQUIRED)
include_directories("${CGAL_DIR}/include")

find_package(GMP REQUIRED)
include_directories("${GMP_INCLUDE_DIR}")

find_package(MPFR REQUIRED)

AddCompilerDefinitions("SOFA_BUILD_CGALPLUGIN")
AddLinkerDependencies(SofaGuiQt ${Boost_LIBRARIES} ${GMP_LIBRARIES} ${MPFR_LIBRARIES} ${CGAL_LIBRARIES})

include(${SOFA_CMAKE_DIR}/postProject.cmake)