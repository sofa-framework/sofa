cmake_minimum_required(VERSION 2.8.12)

project(Sofa)

# Enable the organisation in folders for generators that support
# it. (E.g. some versions of Visual Studio have 'solution folders')
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if( CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
    set( CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install CACHE PATH "Install path prefix, prepended onto install directories." FORCE )
endif (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )

message("Install prefix: ${CMAKE_INSTALL_PREFIX}")

# Set the output directories globally
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)



# Environment
set(SOFA_CMAKE_DIR "${Sofa_SOURCE_DIR}/cmake" )
list(APPEND CMAKE_MODULE_PATH "${SOFA_CMAKE_DIR}/Modules")
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})

include(CMakeSofaBuildFlags)



# Sofa libraries
set(SOFA_EXTERNALS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/sofa-win-dependencies-21-11-2013")
set(SOFA_FRAMEWORK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/SofaFramework")
set(SOFA_SIMULATION_DIR "${CMAKE_CURRENT_SOURCE_DIR}/SofaSimulation")
set(SOFA_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/SofaBase")
set(SOFA_COMMON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/SofaCommon")
set(SOFA_GENERAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/SofaGeneral")
set(SOFA_ADVANCED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/SofaAdvanced")
set(SOFA_MISC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/SofaMisc")
set(SOFA_GUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/SofaGui")
set(SOFA_PLUGINS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/SofaPlugins")
set(SOFA_EXAMPLES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/examples")
set(SOFA_SHARE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/share")

# cmake functions specific to sofa that needs to be redistributed.
add_subdirectory( ${SOFA_CMAKE_DIR} )

# Dependency pack for Windows
if(MSVC)
    if(EXISTS ${SOFA_EXTERNALS_DIR} )
        list(APPEND CMAKE_INCLUDE_PATH ${SOFA_EXTERNALS_DIR}/include)
        if(CMAKE_CL_64)
            list(APPEND CMAKE_LIBRARY_PATH ${SOFA_EXTERNALS_DIR}/lib/win64)
        else()
            list(APPEND CMAKE_LIBRARY_PATH ${SOFA_EXTERNALS_DIR}/lib/win32)
        endif()
        add_subdirectory("${SOFA_EXTERNALS_DIR}")
    endif()
endif()

# Install resource files
install(DIRECTORY "${SOFA_SHARE_DIR}/" DESTINATION share/sofa)
install(DIRECTORY "${SOFA_EXAMPLES_DIR}/" DESTINATION share/sofa/examples)
# Create etc/sofa.ini: it contains the paths to share/ and examples/. In the
# build directory, it points to the source tree, whereas in the install
# directory, it contains to relative paths to the installed resource directory.
set(SHARE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/share)
set(EXAMPLES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/examples)
configure_file(etc/sofa.ini.in ${CMAKE_BINARY_DIR}/etc/sofa.ini)
set(SHARE_DIR ../share/sofa)
set(EXAMPLES_DIR ../share/sofa/examples)
configure_file(etc/sofa.ini.in ${CMAKE_BINARY_DIR}/etc/installedSofa.ini)

install(FILES ${CMAKE_BINARY_DIR}/etc/installedSofa.ini
        DESTINATION etc
        RENAME sofa.ini)

### Extlibs

option(SOFA_TESTS "Compile the automatic tests for Sofa, along with the gtest library." ON)
if(SOFA_TESTS)
    set(gtest_force_shared_crt ON CACHE BOOL "")
    add_subdirectory(extlibs/gtest)
endif()

## Necessary ones
add_subdirectory(extlibs/SuiteSparse)
add_subdirectory(extlibs/csparse)
set(EIGEN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/extlibs/eigen-3.2.1"
    CACHE PATH "Directory search by find_package(Eigen3)")
add_subdirectory(extlibs/eigen-3.2.1)
add_subdirectory(extlibs/miniFlowVR)
add_subdirectory(extlibs/newmat)
add_subdirectory(extlibs/tinyxml)

## Optional ones
add_subdirectory(extlibs/CImg)
add_subdirectory(extlibs/libQGLViewer-2.4.0/QGLViewer)

option(SOFA_USE_MINIBOOST
"Use the (header-only) subset of Boost distributed in the extlibs/ directory
(a.k.a. miniBoost).  This subset is sufficient to compile the framework and the
default components, but some plugins need other components of Boost." ON)
if(SOFA_USE_MINIBOOST)
    add_subdirectory(extlibs/miniBoost ${CMAKE_CURRENT_BINARY_DIR}/miniBoost)
endif()

option(SOFA_COMPILE_ARTRACK
"Build the ARTrack library distributed in the extlibs/ directory.  This library
is used only by the ARTrack plugin."  OFF)
if(SOFA_COMPILE_ARTRACK)
    add_subdirectory(extlibs/ARTrack ${CMAKE_CURRENT_BINARY_DIR}/ARTrackLib)
endif()

# framework
add_subdirectory("${SOFA_FRAMEWORK_DIR}")

# simulation
add_subdirectory("${SOFA_SIMULATION_DIR}")

# base
add_subdirectory("${SOFA_BASE_DIR}")

# common
add_subdirectory("${SOFA_COMMON_DIR}")

# general
add_subdirectory("${SOFA_GENERAL_DIR}")

# advanced
add_subdirectory("${SOFA_ADVANCED_DIR}")

# misc
add_subdirectory("${SOFA_MISC_DIR}")

# gui
add_subdirectory("${SOFA_GUI_DIR}")

# SofaTest depends on SofaPython, so we add it first.
option(SOFA-PLUGIN_SOFAPYTHON "Build the SofaPython plugin." OFF)
if(SOFA-PLUGIN_SOFAPYTHON)
    add_subdirectory(applications/plugins/SofaPython ${CMAKE_CURRENT_BINARY_DIR}/SofaPlugins/SofaPython)
    set_target_properties(SofaPython PROPERTIES FOLDER "SofaPlugins") # IDE folder
endif()

if(SOFA_TESTS)
    # Library used by SofaTest.
    add_subdirectory(applications/plugins/SceneCreator)
    # Library used to write high level tests involving many components.
    add_subdirectory(applications/plugins/SofaTest)
    # Tests for all the modules, written using SofaTest.
    add_subdirectory(modules/tests)
endif()

# plugins
add_subdirectory("${SOFA_PLUGINS_DIR}")
