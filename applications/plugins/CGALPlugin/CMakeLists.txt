include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
    DecimateMesh.h
    DecimateMesh.inl
    MeshGenerationFromPolyhedron.h
    MeshGenerationFromPolyhedron.inl
    TriangularConvexHull3D.h
    TriangularConvexHull3D.inl
    initCGALPlugin.h
)

set(SOURCE_FILES
    DecimateMesh.cpp
    MeshGenerationFromPolyhedron.cpp
    TriangularConvexHull3D.cpp
    initCGALPlugin.cpp
)

set(README_FILES CGALPlugin.txt)

if(WIN32)

    #copy required CGAL dlls
    file(GLOB sharedObjects "${CGAL_DIR}/bin/*.dll")
    foreach(sharedObject ${sharedObjects})
        file(COPY ${sharedObject} DESTINATION "${SOFA_BIN_DIR}")
    endforeach()

endif()

#dependency to FULL boost libraries....
if(NOT SOFA-EXTERNAL_BOOST)
    message(FATAL_ERROR "CGALPlugin needs full boost. Please modify the SOFA-EXTERNAL_BOOST_PATH option to fix it...")
endif()

find_package(CGAL REQUIRED)
include_directories("${CGAL_DIR}/include")

find_package(GMP REQUIRED)
include_directories("${GMP_INCLUDE_DIR}")

find_package(MPFR REQUIRED)

if(SOFA-PLUGIN_IMAGE)
    list(APPEND HEADER_FILES MeshGenerationFromImage.h)
    list(APPEND HEADER_FILES MeshGenerationFromImage.inl)
    list(APPEND SOURCE_FILES MeshGenerationFromImage.cpp)
    AddLinkerDependencies(${CGAL_ImageIO_LIBRARY} image)
endif()

AddCompilerDefinitions("SOFA_BUILD_CGALPLUGIN")
AddLinkerDependencies(SofaGuiQt ${Boost_LIBRARIES} ${GMP_LIBRARIES} ${MPFR_LIBRARIES} ${CGAL_LIBRARY})
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})


include(${SOFA_CMAKE_DIR}/postProject.cmake)
