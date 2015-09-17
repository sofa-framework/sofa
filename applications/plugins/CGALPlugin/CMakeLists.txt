cmake_minimum_required(VERSION 2.8.12)
project(CGALPlugin)

set(CGALPLUGIN_MAJOR_VERSION 0)
set(CGALPLUGIN_MINOR_VERSION 1)
set(CGALPLUGIN_VERSION ${CGALPLUGIN_MAJOR_VERSION}.${CGALPLUGIN_MINOR_VERSION})

set(HEADER_FILES
    DecimateMesh.h
    DecimateMesh.inl
    MeshGenerationFromPolyhedron.h
    MeshGenerationFromPolyhedron.inl
    TriangularConvexHull3D.h
    TriangularConvexHull3D.inl
)

set(SOURCE_FILES
    DecimateMesh.cpp
    MeshGenerationFromPolyhedron.cpp
    TriangularConvexHull3D.cpp
    initCGALPlugin.cpp
)

set(README_FILES CGALPlugin.txt)


### Dependencies


find_package(SofaFramework REQUIRED)
find_package(CGAL REQUIRED)
find_package(image QUIET)
if(image_FOUND)
    list(APPEND HEADER_FILES MeshGenerationFromImage.h)
    list(APPEND HEADER_FILES MeshGenerationFromImage.inl)
    list(APPEND SOURCE_FILES MeshGenerationFromImage.cpp)
else()
    message(STATUS "CGALPlugin: could not find image, won't build MeshGenerationFromImage")
endif()
## CGAL dependencies
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
find_package(MPFR REQUIRED)
find_package(GMP REQUIRED)
find_package(Boost QUIET REQUIRED COMPONENTS system thread)


## CGALPlugin/config.h
configure_file(config.h.in "${CMAKE_BINARY_DIR}/include/CGALPlugin/config.h")
install(FILES "${CMAKE_BINARY_DIR}/include/CGALPlugin/config.h" DESTINATION "include/CGALPlugin")

if(WIN32)
    # Copy CGAL dlls
    file(GLOB sharedObjects "${CGAL_DIR}/bin/*.dll")
    foreach(sharedObject ${sharedObjects})
        file(COPY ${sharedObject} DESTINATION "${SOFA_BIN_DIR}")
    endforeach()
endif()


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")
target_include_directories(${PROJECT_NAME} PUBLIC ${GMP_INCLUDE_DIR} ${MPFR_INCLUDE_DIR} ${CGAL_DIR}/include)
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${CGALPLUGIN_VERSION})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_CGALPLUGIN")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")
target_link_libraries(${PROJECT_NAME} SofaCore ${Boost_LIBRARIES} ${GMP_LIBRARIES} ${MPFR_LIBRARIES} ${CGAL_LIBRARY})
if(image_FOUND)
    target_link_libraries(${PROJECT_NAME} ${CGAL_ImageIO_LIBRARY} image)
endif()

## Install rules for the library and the headers; CMake package configurations files
include(SofaPackageConfigHelpers)
sofa_create_package(CGALPlugin ${CGALPLUGIN_VERSION} CGALPlugin CGALPlugin)

## Install rules for the resources
install(DIRECTORY scenes/ DESTINATION share/sofa/plugins/${PROJECT_NAME})
