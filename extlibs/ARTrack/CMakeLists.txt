cmake_minimum_required(VERSION 3.1)
project(ARTrackLib)

set(HEADER_FILES dtracklib.h)
set(SOURCE_FILES mainTracker.cpp)

add_library(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "SofaExtlibs")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

set(ARTRACK_COMPILER_FLAGS "-Duse_namespace")

# The code must be relocatable if we want to link a shared library against it
if("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xGNU" OR "x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xClang")
    set(ARTRACK_COMPILER_FLAGS "${ARTRACK_COMPILER_FLAGS} -fPIC")
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS ${ARTRACK_COMPILER_FLAGS})

include(${SOFA_KERNEL_SOURCE_DIR}/SofaFramework/SofaMacros.cmake)
sofa_create_package(ARTrackLib 0.0 ARTrackLib "")
