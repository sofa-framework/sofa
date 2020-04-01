cmake_minimum_required(VERSION 3.1)
project(CGALPlugin VERSION 0.1)

set(PLUGIN_CGAL_SRC_DIR src/CGALPlugin)
set(HEADER_FILES
    ${PLUGIN_CGAL_SRC_DIR}/config.h.in
    ${PLUGIN_CGAL_SRC_DIR}/DecimateMesh.h
    ${PLUGIN_CGAL_SRC_DIR}/DecimateMesh.inl
    ${PLUGIN_CGAL_SRC_DIR}/MeshGenerationFromPolyhedron.h
    ${PLUGIN_CGAL_SRC_DIR}/MeshGenerationFromPolyhedron.inl
    ${PLUGIN_CGAL_SRC_DIR}/TriangularConvexHull3D.h
    ${PLUGIN_CGAL_SRC_DIR}/TriangularConvexHull3D.inl
    ${PLUGIN_CGAL_SRC_DIR}/CylinderMesh.h
    ${PLUGIN_CGAL_SRC_DIR}/CylinderMesh.inl
    ${PLUGIN_CGAL_SRC_DIR}/Refine2DMesh.h
    ${PLUGIN_CGAL_SRC_DIR}/Refine2DMesh.inl
)

set(SOURCE_FILES
    ${PLUGIN_CGAL_SRC_DIR}/initCGALPlugin.cpp
    ${PLUGIN_CGAL_SRC_DIR}/DecimateMesh.cpp
    ${PLUGIN_CGAL_SRC_DIR}/MeshGenerationFromPolyhedron.cpp
    ${PLUGIN_CGAL_SRC_DIR}/TriangularConvexHull3D.cpp
    ${PLUGIN_CGAL_SRC_DIR}/CylinderMesh.cpp
    ${PLUGIN_CGAL_SRC_DIR}/Refine2DMesh.cpp
)

set(README_FILES CGALPlugin.txt)


# Add CGAL dependencies
find_package(SofaFramework REQUIRED)

# Check if CImg plugin is build, If yes add more files
find_package(image QUIET)
if(image_FOUND)
    list(APPEND HEADER_FILES ${PLUGIN_CGAL_SRC_DIR}/MeshGenerationFromImage.h)
    list(APPEND HEADER_FILES ${PLUGIN_CGAL_SRC_DIR}/MeshGenerationFromImage.inl)
    list(APPEND SOURCE_FILES ${PLUGIN_CGAL_SRC_DIR}/MeshGenerationFromImage.cpp)
    
    find_package(CGAL REQUIRED COMPONENTS ImageIO)
else()
    find_package(CGAL REQUIRED)
endif()

## CGAL dependencies
find_package(MPFR REQUIRED)
find_package(GMP REQUIRED)
#find_package(Boost QUIET COMPONENTS system thread)

# Get CGAL version
message(STATUS "CGAL VERSION ${CGAL_MAJOR_VERSION}.${CGAL_MINOR_VERSION}")
set(CGAL_VERSION ${CGAL_MAJOR_VERSION}.${CGAL_MINOR_VERSION})
if (CGAL_VERSION VERSION_GREATER 4.9) # if CGAL >= 4.10
    add_definitions(-DCGAL_MESH_3_VERBOSE=0)
endif()


if(WIN32)
#    sofa_copy_libraries(${GMP_LIBRARIES} ${MPFR_LIBRARIES} ${CGAL_LIBRARIES})
#    sofa_install_libraries(${GMP_LIBRARIES} ${MPFR_LIBRARIES} ${CGAL_LIBRARIES})
endif()

# Create the plugin library.
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

# Set define dllimport/dllexport mechanism on Windows.
target_compile_definitions(${PROJECT_NAME} PRIVATE "-DSOFA_BUILD_CGALPLUGIN")

# Link the plugin library to its dependencies (other libraries).
target_link_libraries(${PROJECT_NAME} PUBLIC SofaCore SofaSimulationCommon ${Boost_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${GMP_LIBRARIES} ${MPFR_LIBRARIES} CGAL)

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")
target_include_directories(${PROJECT_NAME} PUBLIC ${GMP_INCLUDE_DIR} ${MPFR_INCLUDE_DIR} ${CGAL_INCLUDE_DIRS})

if(CImgPlugin_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${CGAL_ImageIO_LIBRARY} image)
    #target_include_directories(${PROJECT_NAME} PUBLIC ${CImg_INCLUDE_DIRS})
else()
    target_link_libraries(${PROJECT_NAME} ${CGAL_ImageIO_LIBRARY})
endif()


if(SOFA_BUILD_TESTS)
    find_package(SofaTest QUIET)
    if(SofaTest_FOUND)
        add_subdirectory(CGALPlugin_test)
    endif()
endif()

# Install rules for the library and the headers; CMake package configurations files
sofa_generate_package(
    NAME ${PROJECT_NAME}
    TARGETS ${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    INCLUDE_SOURCE_DIR "src/CGALPlugin"
    INCLUDE_INSTALL_DIR "CGALPlugin"
    RELOCATABLE "plugins"
    )
