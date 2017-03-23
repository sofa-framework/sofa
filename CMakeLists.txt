cmake_minimum_required(VERSION 3.1)
project(Sofa)

set(SOFA_KERNEL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/SofaKernel" CACHE STRING "Path to SofaKernel")

if(NOT EXISTS ${SOFA_KERNEL_SOURCE_DIR}/framework)
    set(SOFA_KERNEL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/SofaKernel")
    if(NOT EXISTS ${SOFA_KERNEL_SOURCE_DIR}/framework)
        message( FATAL_ERROR "SofaKernel is not present or is inaccessible. Check if ${SOFA_KERNEL_SOURCE_DIR} exists or is accessible.")
    endif()
endif()

## Default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

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

## Change default install prefix
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install CACHE PATH "Install path prefix, prepended onto install directories." FORCE)
endif()
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")

# Remove generated CMake files, this prevents CMake from finding packages that
# were disabled (like, unchecked in cmake-gui) after being built once.
file(REMOVE_RECURSE ${CMAKE_BINARY_DIR}/cmake)
# Remove generated SofaPython configuration files, to prevent SofaPython from
# adding paths to disabled plugins.
file(REMOVE_RECURSE ${CMAKE_BINARY_DIR}/etc/sofa/python.d)

set(ARCHIVE_OUTPUT_DIRECTORY lib)
set(RUNTIME_OUTPUT_DIRECTORY bin)

if(WIN32)
	set(LIBRARY_OUTPUT_DIRECTORY ${RUNTIME_OUTPUT_DIRECTORY})
else()
	set(LIBRARY_OUTPUT_DIRECTORY ${ARCHIVE_OUTPUT_DIRECTORY})
endif()

## Set the output directories globally
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${ARCHIVE_OUTPUT_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${RUNTIME_OUTPUT_DIRECTORY})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${LIBRARY_OUTPUT_DIRECTORY})

## Environment
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")
list(APPEND CMAKE_MODULE_PATH "${SOFA_KERNEL_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${SOFA_KERNEL_SOURCE_DIR}/cmake/Modules")
list(APPEND CMAKE_MODULE_PATH "${SOFA_KERNEL_SOURCE_DIR}/SofaFramework")
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR}/extlibs)
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR}/applications/plugins)
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR}/applications/projects)

## RPATH
if(UNIX)
    # RPATH is a field in ELF binaries that is used as a hint by the system
    # loader to find needed shared libraries.
    #
    # In the build directory, cmake creates binaries with absolute paths in
    # RPATH.  And by default, it strips RPATH from installed binaries.  Here we
    # use CMAKE_INSTALL_RPATH to set a relative RPATH.  By doing so, we avoid
    # the need to play with LD_LIBRARY_PATH to get applications to run.
    set(CMAKE_INSTALL_RPATH "../lib")

    if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        set(CMAKE_MACOSX_RPATH ON)
        
    endif()

endif(UNIX)

include(CompilerOptions)


## Boost is now mandatory.
find_package(Boost QUIET)
if(NOT Boost_FOUND)
    message("Did not find a boost directory from Boost_LIBRARY_DIR and Boost_INCLUDE_DIR, will try to search one in possible directories")
    ## Try to find an installed boost.
    ## (only for Windows ; we suppose boost in a POSIX way on an Unix system)
    ## It will check from different Mount Points, different subdirectory.
    ## For now, it is only searching within one level of the file hierarchy.
    ## We assume also that the boost root dirname contains boost....
    ## TODO: Set an other hint from the PATH, as the user should set it with the binaries directory when he launches Sofa
    if(WIN32)
        #set roots
        set(BOOST_TEST_ROOTS 
            "C:" "D:" "E:" "F:" "G:" "H:"
            )
        #set credible subdirectories
        set(BOOST_TEST_SUBDIRECTORIES 
            "/" "/local" "/Dependencies" "/Program Files" "/Program Files (x86)"
            )

        #create all combinations
        set(BOOST_TEST_DIRECTORIES)
        foreach( BOOST_TEST_ROOT ${BOOST_TEST_ROOTS} )
            foreach( BOOST_TEST_SUBDIRECTORY ${BOOST_TEST_SUBDIRECTORIES} )
                list(APPEND BOOST_TEST_DIRECTORIES ${BOOST_TEST_ROOT}${BOOST_TEST_SUBDIRECTORY}/)
            endforeach()
        endforeach()

        #test if contains a boost_something directory
        set(BOOST_TEST_POTENTIAL_DIRECTORIES)
        foreach( BOOST_TEST_DIRECTORY ${BOOST_TEST_DIRECTORIES} )
            file(GLOB BOOST_TEST_TEMP "${BOOST_TEST_DIRECTORY}*boost*" )
            list(APPEND BOOST_TEST_POTENTIAL_DIRECTORIES "${BOOST_TEST_TEMP}" )
        endforeach()

        ## BOOST includes
        #test if this directory contain a directory with the expected includes
        #(boost_include_dir = <some_directory>/boost/....h)
        find_path (BOOST_TEST_PATH boost/config.hpp
            PATHS ${BOOST_TEST_POTENTIAL_DIRECTORIES}
            )

        #remove the test variable from the cache
        SET(BOOST_TEST_PATH ${BOOST_TEST_PATH} CACHE INTERNAL "string")

        if(NOT BOOST_TEST_PATH)
            message(FATAL_ERROR "Boost includes was not found automatically, please set Boost_INCLUDE_DIR (i.e consistent with this: <BOOST_INCLUDE_DIR>/boost/thread/thread.hpp.")
        else()
            set(Boost_INCLUDE_DIR ${BOOST_TEST_PATH})
            message("Found boost includes at ${Boost_INCLUDE_DIR}")

            ## BOOST libraries
            #test if this directory contain a directory with the expected library
            # recursive = overkill ?
            file(GLOB_RECURSE BOOST_TEST_PATH2  "${Boost_INCLUDE_DIR}/*/boost_thread-vc???-mt-?_??.lib")

            SET(BOOST_TEST_PATH2 ${BOOST_TEST_PATH2} CACHE INTERNAL "string")
            if(NOT BOOST_TEST_PATH2)
                message("Boost libraries were not found automatically, please set Boost_LIBRARY_DIR.")
            else()
                set(Boost_THREAD_LIBRARY_RELEASE ${BOOST_TEST_PATH2})
                message("Found boost libs at ${Boost_THREAD_LIBRARY_RELEASE}")
            endif()
        endif()
    endif()
endif()

find_package(Boost REQUIRED)

### Dependency pack for Windows
if(MSVC)
    #define BOOST_ALL_DYN_LINK needed for dynamic linking with boost libraries
    add_definitions(-DBOOST_ALL_DYN_LINK)

    set(SOFA_DEPENDENCY_PACK_DIR "" CACHE PATH "sofa windows dependency pack path")
    if(SOFA_DEPENDENCY_PACK_DIR)
        set(DEPENDENCY_PACK_DIR "${SOFA_DEPENDENCY_PACK_DIR}")
    else()
        # Default
        set(DEPENDENCY_PACK_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    endif()

    list(APPEND CMAKE_INCLUDE_PATH ${DEPENDENCY_PACK_DIR}/include)
    if(CMAKE_CL_64)
        list(APPEND CMAKE_LIBRARY_PATH ${DEPENDENCY_PACK_DIR}/lib/win64)
    else()
        list(APPEND CMAKE_LIBRARY_PATH ${DEPENDENCY_PACK_DIR}/lib/win32)
    endif()

    if(CMAKE_CL_64)
        file(GLOB DLLS "${DEPENDENCY_PACK_DIR}/lib/win64/*.dll")
        file(GLOB LIBS "${DEPENDENCY_PACK_DIR}/lib/win64/*.lib")
    else()
        file(GLOB DLLS "${DEPENDENCY_PACK_DIR}/lib/win32/*.dll")
        file(GLOB LIBS "${DEPENDENCY_PACK_DIR}/lib/win32/*.lib")
    endif()
    ## Copy DLLs to the build tree
    if(CMAKE_CONFIGURATION_TYPES) # Multi-config generator (MSVC)
        foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
            file(COPY ${DLLS} DESTINATION "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CONFIG}")
        endforeach()
    else()                      # Single-config generator (nmake)
        file(COPY ${DLLS} DESTINATION "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
    endif()
    ## Install DLLs
    install(FILES ${DLLS} DESTINATION ${LIBRARY_OUTPUT_DIRECTORY})
    install(FILES ${LIBS} DESTINATION ${ARCHIVE_OUTPUT_DIRECTORY})
    install(DIRECTORY ${DEPENDENCY_PACK_DIR}/include/ DESTINATION include)
endif()

option(SOFA_INSTALL_RESOURCES_FILES "Copy resources files (etc/, share/, examples/) when installing" ON)
if(SOFA_INSTALL_RESOURCES_FILES)
    ## Install resource files
    install(DIRECTORY share/ DESTINATION share/sofa)
    install(DIRECTORY examples/ DESTINATION share/sofa/examples)
    # Create etc/sofa.ini: it contains the paths to share/ and examples/. In the
    # build directory, it points to the source tree, whereas in the install
    # directory, it contains to relative paths to the installed resource directory.
    set(SHARE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/share")
    set(EXAMPLES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/examples")
    configure_file(${SOFA_KERNEL_SOURCE_DIR}/etc/sofa.ini.in "${CMAKE_BINARY_DIR}/etc/sofa.ini")
    set(SHARE_DIR "../share/sofa")
    set(EXAMPLES_DIR "../share/sofa/examples")
    configure_file(${SOFA_KERNEL_SOURCE_DIR}/etc/sofa.ini.in "${CMAKE_BINARY_DIR}/etc/installedSofa.ini")
    install(FILES "${CMAKE_BINARY_DIR}/etc/installedSofa.ini" DESTINATION etc RENAME sofa.ini)
endif()

# Temporary (todo: think about this typedef thing)
install(DIRECTORY modules/sofa/component/typedef/ DESTINATION include/sofa/component/typedef)


### Mask
option(SOFA_USE_MASK "Use mask optimization" ON)


### Testing
option(SOFA_BUILD_TESTS "Compile the automatic tests for Sofa, along with the gtest library." ON)
if(SOFA_BUILD_TESTS)
    # Enable testing features of cmake, like the add_test() command.
    enable_testing()
    add_subdirectory(extlibs/gtest)
endif()

### Extlibs
add_subdirectory(extlibs)


### Main Sofa subdirectories

add_subdirectory(${SOFA_KERNEL_SOURCE_DIR}/SofaFramework)
add_subdirectory(${SOFA_KERNEL_SOURCE_DIR}/SofaSimulation)
if(SOFA_BUILD_TESTS)
    # Depends on SofaSimulation
    add_subdirectory(tools/SofaGTestMain)
endif()
add_subdirectory(${SOFA_KERNEL_SOURCE_DIR}/SofaBase)
add_subdirectory(${SOFA_KERNEL_SOURCE_DIR}/SofaCommon)
add_subdirectory(SofaGeneral)
add_subdirectory(SofaAdvanced)
add_subdirectory(SofaMisc)
add_subdirectory(SofaGui)

# SofaTest depends on SofaPython, so we add SofaPython before SofaTest.
sofa_add_plugin( applications/plugins/SofaPython SofaPython )


## Scene creator option
option(SOFA_BUILD_SCENECREATOR "Compile the C++ API located in applications/plugins/SceneCreator" OFF)

## Tutorials option
option(SOFA_BUILD_TUTORIALS "Build (most of) the tutorials available." OFF)


# Activate SceneCreator when required
if(SOFA_BUILD_TESTS OR SOFA_BUILD_TUTORIALS)
    set(SOFA_BUILD_SCENECREATOR ON CACHE BOOL "" FORCE)
endif()

# Library used by SOFA_BUILD_TESTS and SOFA_BUILD_TUTORIALS
if(SOFA_BUILD_SCENECREATOR)
    add_subdirectory(applications/plugins/SceneCreator SceneCreator)
endif()

# Tests : define subdirectories
if(SOFA_BUILD_TESTS)
    # Library used to write high level tests involving many components.
    add_subdirectory(applications/plugins/SofaTest SofaTest)
    # Tests for all the modules, written using SofaTest.
    add_subdirectory(modules/tests)
    # SofaPython_test is a special case, because it depends on
    # SofaTest, which depends on SofaPython...
    if(PLUGIN_SOFAPYTHON)
        add_subdirectory(applications/plugins/SofaPython/SofaPython_test)
    endif()

    # Not sure what is it
    add_subdirectory(applications/plugins/SofaTest/SofaTest_test SofaTest/SofaTest_test)
endif()

## Plugins
add_subdirectory(applications/plugins)

## Applications
add_subdirectory(applications/projects)

# Tutorial add subdirectory
if(SOFA_BUILD_TUTORIALS)
    add_subdirectory(applications/tutorials)
endif()

## Build external projects at the same time
set(SOFA_EXTERNAL_DIRECTORIES "" CACHE STRING "list of paths separated by ';'")
if(NOT "${SOFA_EXTERNAL_DIRECTORIES}" STREQUAL "")
    foreach(dir ${SOFA_EXTERNAL_DIRECTORIES})
        get_filename_component(name ${dir} NAME) # Get the name of the actual directory
        message("Adding external directory: ${name} (${dir})")
        add_subdirectory(${dir} "${CMAKE_CURRENT_BINARY_DIR}/external_directories/${name}")
    endforeach()
endif()


## Custom
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/custom.cmake")
    message("Adding custom file")
    include( "custom.cmake" )
endif()


## when installing, keep an eye on options/params/sources used for compilation
## this should be internal and not delivered, but this is definitively useful
sofa_install_git_version( "sofa" ${CMAKE_CURRENT_SOURCE_DIR} )
install(FILES "${CMAKE_BINARY_DIR}/CMakeCache.txt" DESTINATION .)
install(FILES "${CMAKE_SOURCE_DIR}/README.md" DESTINATION .)
install(FILES "${CMAKE_SOURCE_DIR}/CHANGELOG.md" DESTINATION .)
install(FILES "${CMAKE_SOURCE_DIR}/LICENSE.LGPL.txt" DESTINATION .)
install(FILES "${CMAKE_SOURCE_DIR}/Authors.txt" DESTINATION .)

#CPack install
SET(CPACK_PACKAGE_VERSION "17.dev.0")
SET(CPACK_PACKAGE_VERSION_MAJOR "17")
SET(CPACK_PACKAGE_VERSION_MINOR "dev")
SET(CPACK_PACKAGE_VERSION_PATCH "0")

#dirty hack to pack component we want (named BundlePack from runSofa for example, and that will install .app + share)
#if not set, it will install everything as usual
#TODO: Redesign of a better architecture about bundling (runSofa ? libraries ? modeler ? other application ?)
get_property(TEMP_CPACK_COMPONENTS_ALL GLOBAL PROPERTY RUNSOFA_CPACK_COMPONENTS_ALL )
if(TEMP_CPACK_COMPONENTS_ALL)
    set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
    set(CPACK_COMPONENTS_ALL ${TEMP_CPACK_COMPONENTS_ALL})
endif()

INCLUDE(CPack)
