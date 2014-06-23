cmake_minimum_required(VERSION 2.8.8)

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


set(SOFA_FRAMEWORK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/framework")
set(SOFA_MODULES_DIR   "${CMAKE_CURRENT_SOURCE_DIR}/modules")
set(SOFA_SIMULATION_DIR "${SOFA_MODULES_DIR}/sofa/simulation")
set(SOFA_EXTLIBS_DIR   "${CMAKE_CURRENT_SOURCE_DIR}/extlibs")

# extlibs
set(SOFA_PROJECT_FOLDER "Extlibs")
add_subdirectory(${SOFA_EXTLIBS_DIR}/tinyxml)
add_subdirectory(${SOFA_EXTLIBS_DIR}/gtest)

# framework
set(SOFA_PROJECT_FOLDER "SofaFramework")
add_subdirectory("${SOFA_FRAMEWORK_DIR}")

# simulation
set(SOFA_PROJECT_FOLDER "SofaSimulation")
add_subdirectory("${SOFA_SIMULATION_DIR}")

