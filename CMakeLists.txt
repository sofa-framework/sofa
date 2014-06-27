cmake_minimum_required(VERSION 2.8.12)

set(SOLUTION_NAME "Sofa")
project(${SOLUTION_NAME})

# MSVC: Use solution folders.
if(MSVC)
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif()
if( CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
  set( CMAKE_INSTALL_PREFIX ${Sofa_SOURCE_DIR}/install CACHE PATH "Install path prefix, prepended onto install directories." FORCE )
endif (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )

# Set the output directories globally
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Environment
set(SOFA_CMAKE_DIR "${Sofa_SOURCE_DIR}/cmake" )
list(APPEND CMAKE_MODULE_PATH "${SOFA_CMAKE_DIR}/Modules")
include("${SOFA_CMAKE_DIR}/SofaBuildFlags.cmake")

list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})


set(SOFA_DEPS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/SofaDeps")
set(SOFA_FRAMEWORK_DIR   "${CMAKE_CURRENT_SOURCE_DIR}/SofaFramework")
set(SOFA_SIMULATION_DIR "${CMAKE_CURRENT_SOURCE_DIR}/SofaSimulation")
set(SOFA_BASE_DIR   "${CMAKE_CURRENT_SOURCE_DIR}/SofaBase"  )
set(SOFA_COMMON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/SofaCommon")

# deps
add_subdirectory("${SOFA_DEPS_DIR}")

# framework
add_subdirectory("${SOFA_FRAMEWORK_DIR}")

# simulation
add_subdirectory("${SOFA_SIMULATION_DIR}")

# base
add_subdirectory("${SOFA_BASE_DIR}")

# common
add_subdirectory("${SOFA_COMMON_DIR}")

