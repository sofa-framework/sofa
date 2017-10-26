cmake_minimum_required(VERSION 3.1)
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
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

include(${SOFA_KERNEL_SOURCE_DIR}/SofaFramework/SofaMacros.cmake)
sofa_create_package(CSparse 0.0 ${PROJECT_NAME} "")
