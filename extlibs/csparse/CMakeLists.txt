cmake_minimum_required(VERSION 2.8.12)
project(csparse)

set(HEADER_FILES
    csparse.h
    ldl.h
    UFconfig.h)

set(SOURCE_FILES
    csparse.c
    ldl.c)

# The code must be relocatable if we want to link a shared library against it.
if("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xGNU" OR "x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xClang")
    add_compile_options(-fPIC)
endif()

add_library(${PROJECT_NAME} STATIC ${HEADER_FILES} ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "SofaExtlibs")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

include(../../SofaFramework/SofaPackageConfigHelpers.cmake)
sofa_create_package(CSparse 0.0 csparse "")
