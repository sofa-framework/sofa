cmake_minimum_required(VERSION 2.8.12)
project(ARTrackLib)

set(HEADER_FILES dtracklib.h)
set(SOURCE_FILES mainTracker.cpp)

add_library(ARTrackLib ${HEADER_FILES} ${SOURCE_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "SofaExtlibs")
target_include_directories(ARTrackLib PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
set_target_properties(ARTrackLib PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

set(ARTRACK_COMPILER_FLAGS "-Duse_namespace")

# The code must be relocatable if we want to link a shared library against it
if("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xGNU" OR "x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xClang")
    set(ARTRACK_COMPILER_FLAGS "${ARTRACK_COMPILER_FLAGS} -fPIC")
endif()

set_target_properties(ARTrackLib PROPERTIES COMPILE_FLAGS ${ARTRACK_COMPILER_FLAGS})

include(../../SofaFramework/SofaPackageConfigHelpers.cmake)
sofa_create_package(ARTrackLib 0.0 ARTrackLib "")
