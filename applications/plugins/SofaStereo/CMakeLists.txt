cmake_minimum_required(VERSION 3.1)
project(SofaStereo)

set(SOFASTEREO_MAJOR_VERSION 0)
set(SOFASTEREO_MINOR_VERSION 1)
set(SOFASTEREO_VERSION ${IMAGE_MAJOR_VERSION}.${IMAGE_MINOR_VERSION})

## RPATH
if(UNIX)
    set(CMAKE_INSTALL_RPATH "../lib")
    if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        set(CMAKE_MACOSX_RPATH ON)
    endif()
endif(UNIX)

set(HEADER_FILES
    StereoOglModel.h
    StereoCamera.h
    initPlugin.h
)

set(SOURCE_FILES
    StereoOglModel.cpp
    StereoCamera.cpp
    initPlugin.cpp
)

find_package(SofaFramework REQUIRED)
find_package(SofaGeneral REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_PLUGIN_STEREO")
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${SOFASTEREO_VERSION})
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")
target_link_libraries(${PROJECT_NAME} SofaCore SofaOpenglVisual)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")

## Install rules for the library and headers; CMake package configurations files
sofa_create_package(SofaStereo ${SOFASTEREO_VERSION} ${PROJECT_NAME} SofaStereo)

