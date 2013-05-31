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

include_directories("${SOFA_EXTLIBS_DIR}/CGAL")
find_library(CGAL_LIBRARIES NAMES CGAL cgal HINTS "${SOFA_EXTLIBS_DIR}/CGAL")

include_directories("${SOFA_EXTLIBS_DIR}/gmp")
find_library(GMP_LIBRARIES NAMES gmp HINTS "${SOFA_EXTLIBS_DIR}/gmp)

include_directories("${SOFA_EXTLIBS_DIR}/mpfr")
find_library(MPRF_LIBRARIES NAMES mpfr HINTS "${SOFA_EXTLIBS_DIR}/mpfr)

AddCompilerDefinitions("SOFA_BUILD_CGALPLUGIN")
AddLinkerDependencies(SofaGuiQt ${Boost_LIBRARIES} ${CGAL_LIBRARIES} ${GMP_LIBRARIES} ${MPRF_LIBRARIES})

include(${SOFA_CMAKE_DIR}/postProject.cmake)