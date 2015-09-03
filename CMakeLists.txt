cmake_minimum_required(VERSION 2.8.12)

project(Sofa)

## Set some policies to avoid warnings from CMake.
cmake_policy(SET CMP0015 OLD)   # CMake 3.0.2 warns if this is not set.
if(CMAKE_VERSION GREATER 3.0)
    cmake_policy(SET CMP0039 OLD)   # CMake 3.0.2 warns if this is not set.
    cmake_policy(SET CMP0043 OLD)   # CMake 3.2.3 warns if this is not set.
endif()
if(CMAKE_VERSION GREATER 3.1)
    cmake_policy(SET CMP0054 OLD)   # CMake 3.2.3 warns if this is not set.
endif()

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
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR}/extlibs)
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR}/applications/plugins)


include(CMakeSofaBuildFlags)

# C++11 Support
#
# Note for the future: CMake 3.1 has a CMAKE_CXX_STANDARD variable
# that handles versions of the C++ standard...
if(NOT WIN32)
    option(SOFA_USE_C++11
"Use C++11 as a target language. Even though Sofa is written in C++98, it is
compatible with C++11. Targetting this standard allows you to benefit from
a more optimized STL."  OFF)
    if(SOFA_USE_C++11)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11"
            CACHE STRING "Flags used by the compiler during all build types." FORCE)
    endif()
endif()


# cmake functions specific to sofa that needs to be redistributed.
install(FILES cmake/Modules/CMakeSofaBuildFlags.cmake DESTINATION share/cmake/Modules)

# Dependency pack for Windows
if(MSVC)
    set(DEPENDENCY_PACK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/sofa-win-dependencies-21-11-2013")
    if(EXISTS ${DEPENDENCY_PACK_DIR} )
        list(APPEND CMAKE_INCLUDE_PATH ${DEPENDENCY_PACK_DIR}/include)
        if(CMAKE_CL_64)
            list(APPEND CMAKE_LIBRARY_PATH ${DEPENDENCY_PACK_DIR}/lib/win64)
        else()
            list(APPEND CMAKE_LIBRARY_PATH ${DEPENDENCY_PACK_DIR}/lib/win32)
        endif()
        add_subdirectory("${DEPENDENCY_PACK_DIR}")
    endif()
endif()

# Install resource files
install(DIRECTORY share DESTINATION share/sofa)
install(DIRECTORY examples DESTINATION share/sofa/examples)
# Create etc/sofa.ini: it contains the paths to share/ and examples/. In the
# build directory, it points to the source tree, whereas in the install
# directory, it contains to relative paths to the installed resource directory.
set(SHARE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/share")
set(EXAMPLES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/examples")
configure_file(etc/sofa.ini.in "${CMAKE_BINARY_DIR}/etc/sofa.ini")
set(SHARE_DIR "../share/sofa")
set(EXAMPLES_DIR "../share/sofa/examples")
configure_file(etc/sofa.ini.in "${CMAKE_BINARY_DIR}/etc/installedSofa.ini")
install(FILES "${CMAKE_BINARY_DIR}/etc/installedSofa.ini"
        DESTINATION etc
        RENAME sofa.ini)

# Temporary (todo: think about this typedef thing)
install(DIRECTORY modules/sofa/component/typedef/ DESTINATION include/sofa/component/typedef)

### Extlibs

option(SOFA_BUILD_TESTS "Compile the automatic tests for Sofa, along with the gtest library." ON)
if(SOFA_BUILD_TESTS)
    add_subdirectory(extlibs/gtest)
endif()

## Necessary ones
add_subdirectory(extlibs/SuiteSparse)
add_subdirectory(extlibs/csparse)
add_subdirectory(extlibs/eigen-3.2.5 extlibs/eigen)
add_subdirectory(extlibs/miniFlowVR)
add_subdirectory(extlibs/newmat)
add_subdirectory(extlibs/tinyxml)

## Optional ones
add_subdirectory(extlibs/CImg)
add_subdirectory(extlibs/libQGLViewer-2.4.0/QGLViewer extlibs/QGLViewer)

find_package(Boost QUIET)
if(NOT Boost_FOUND)
    add_subdirectory(extlibs/miniBoost)
endif()

option(SOFA_BUILD_ARTRACK
"Build the ARTrack library distributed in the extlibs/ directory.  This library
is used only by the ARTrack plugin."  OFF)
if(SOFA_BUILD_ARTRACK)
    add_subdirectory(extlibs/ARTrack extlibs/ARTrackLib)
endif()

option(SOFA_BUILD_METIS
"Build the metis library distributed in the extlibs/ directory.  It is used only
by the SparseLDLSolver component.  Note that this library is usually available
on Linux distributions." OFF)
if(SOFA_BUILD_METIS)
    add_subdirectory(extlibs/metis-5.1.0 "extlibs/metis")
endif()


add_subdirectory(SofaFramework)
add_subdirectory(SofaSimulation)
add_subdirectory(SofaBase)
add_subdirectory(SofaCommon)
add_subdirectory(SofaGeneral)
add_subdirectory(SofaAdvanced)
add_subdirectory(SofaMisc)
add_subdirectory(SofaGui)

if(SOFA_BUILD_TESTS)
    add_subdirectory(tools/SofaGTestMain)
endif()

# SofaTest depends on SofaPython, so we add SofaPython before SofaTest.
option(PLUGIN_SOFAPYTHON "Build the SofaPython plugin." OFF)
if(PLUGIN_SOFAPYTHON)
    add_subdirectory(applications/plugins/SofaPython)
    set_target_properties(SofaPython PROPERTIES FOLDER "Plugins") # IDE folder
endif()

if(SOFA_BUILD_TESTS)
    # Library used by SofaTest.
    add_subdirectory(applications/plugins/SceneCreator SceneCreator)
    # Library used to write high level tests involving many components.
    add_subdirectory(applications/plugins/SofaTest SofaTest)
    # Tests for all the modules, written using SofaTest.
    add_subdirectory(modules/tests)
endif()

# Plugins
add_subdirectory(applications/plugins)

# Tutorials
option(SOFA_BUILD_TUTORIALS "Build (most of) the tutorials available." OFF)
if(SOFA_BUILD_TUTORIALS)
    add_subdirectory(applications/tutorials)
endif()
