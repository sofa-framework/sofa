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
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR}/applications/packages)
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR}/applications/projects)

# Create etc/sofa.ini: it contains the paths to share/ and examples/. In the
# build directory, it points to the source tree,
set(SHARE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/share")
set(EXAMPLES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/examples")
configure_file(${SOFA_KERNEL_SOURCE_DIR}/etc/sofa.ini.in "${CMAKE_BINARY_DIR}/etc/sofa.ini")


## RPATH
if(UNIX)
    # RPATH is a field in ELF binaries that is used as a hint by the system
    # loader to find needed shared libraries.
    #
    # In the build directory, cmake creates binaries with absolute paths in
    # RPATH.  And by default, it strips RPATH from installed binaries.  Here we
    # use CMAKE_INSTALL_RPATH to set a relative RPATH.  By doing so, we avoid
    # the need to play with LD_LIBRARY_PATH to get applications to run.

    # see https://cmake.org/Wiki/CMake_RPATH_handling for $ORIGIN doc
    set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib:$$ORIGIN/../lib")

    if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        set(CMAKE_MACOSX_RPATH ON)
    endif()
endif(UNIX)

include(CompilerOptions)


## Boost (1.54.0 or higher) is now mandatory.
set(BOOST_MIN_VERSION "1.54.0")
find_package(Boost ${BOOST_MIN_VERSION} REQUIRED)

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

    install(DIRECTORY ${DEPENDENCY_PACK_DIR}/include/ DESTINATION include COMPONENT headers)
    install(DIRECTORY ${DEPENDENCY_PACK_DIR}/licenses/ DESTINATION licenses COMPONENT applications)
endif()

### Mask
option(SOFA_USE_MASK "Use mask optimization" OFF)

### SOFA_DEV_TOOL
option(SOFA_WITH_DEVTOOLS "Compile with developement extra features." ON)

### SOFA_BUILD_PERIMETER
### Sofa using type double or float or both
set(SOFA_BUILD_COMPONENTSET full CACHE STRING
"Compile SOFA with which set of components (core/full/deprecated)
- 'core' for only the core components (only base classes)
- 'light' for stable only component (a minimal set to make simulations)
 - 'standard' for all non-deprecated components (DEFAULT)
 - 'full' for standard set + deprecated components")
set_property(CACHE SOFA_BUILD_COMPONENTSET PROPERTY STRINGS core light standard full)

if(${SOFA_BUILD_COMPONENTSET} STREQUAL core)
    message(STATUS "   Build SOFA with 'core' compenents set")
    set(SOFA_BUILD_COMPONENTSET_LIGHT 0)
    set(SOFA_BUILD_COMPONENTSET_STANDARD 0)
    set(SOFA_BUILD_COMPONENTSET_FULL 0)
    set(SOFAGUI_QQT 0)
elseif(${SOFA_BUILD_COMPONENTSET} STREQUAL light)
    message(STATUS "   Build SOFA with 'light' compenents set")
    set(SOFA_BUILD_COMPONENTSET_LIGHT 1)
    set(SOFA_BUILD_COMPONENTSET_STANDARD 0)
    set(SOFA_BUILD_COMPONENTSET_FULL 0)
    set(SOFAGUI_QQT 0)
elseif(${SOFA_BUILD_COMPONENTSET} STREQUAL standard)
    message(STATUS "   Build SOFA with 'standard' compenents set")
    set(SOFA_BUILD_COMPONENTSET_LIGHT 1)
    set(SOFA_BUILD_COMPONENTSET_STANDARD 1)
    set(SOFA_BUILD_COMPONENTSET_FULL 0)
elseif(${SOFA_BUILD_COMPONENTSET} STREQUAL full)
    message(STATUS "   Build SOFA with 'full' compenents set (incuding deprecated components)")
    set(SOFA_BUILD_COMPONENTSET_LIGHT 1)
    set(SOFA_BUILD_COMPONENTSET_STANDARD 1)
    set(SOFA_BUILD_COMPONENTSET_FULL 1)
else()
    message(STATUS "   Unrecognize option for components set. Possible ones are: core, light, standard and full")
endif()

### Threading
option(SOFA_WITH_THREADING "Compile sofa with thread-safetiness support (PARTIAL/EXPERIMENTAL)" ON)

### Experimental
option(SOFA_WITH_EXPERIMENTAL_FEATURES "Compile sofa with exprimental features regarding sparse matrices treatments under mappings" OFF)

### Testing
add_subdirectory(extlibs/gtest)
option(SOFA_BUILD_TESTS "Compile the automatic tests for Sofa, along with the gtest library." ON)
if(SOFA_BUILD_TESTS)
    # Enable testing features of cmake, like the add_test() command.
    enable_testing()
endif()

### Ninja build pools
option(SOFA_NINJA_BUILD_POOLS "Activate the Ninja build pools feature, to limit the cores used by specific targets" OFF)

### Sofa using type double or float or both
set(SOFA_FLOATING_POINT_TYPE both CACHE STRING
"Type used for floating point values in SOFA. It actually determines:
 - what template instanciations will be compiled (via the definition of the
   SOFA_FLOAT and SOFA_DOUBLE macros)
 - what is the type behind the 'SReal' typedef used throughout SOFA. (If 'both'
   is selected, SReal defaults to double.)")
set_property(CACHE SOFA_FLOATING_POINT_TYPE PROPERTY STRINGS float double both)

if(${SOFA_FLOATING_POINT_TYPE} STREQUAL double)
    set(SOFA_DOUBLE 1)          # Used in sofa/config.h.in
elseif(${SOFA_FLOATING_POINT_TYPE} STREQUAL float)
    set(SOFA_FLOAT 1)           # Used in sofa/config.h.in
endif()

# If you really don't understand the negated logics of SOFA_DOUBLE and
# SOFA_FLOAT please consider using SOFA_WITH_FLOAT and SOFA_WITH_DOUBLE.
# Eg: SOFA_WITH_FLOAT indicate that you need to generate the
# float code and SOFA_WITH_DOUBLE indicates that you
# nedd to generate the double related code.
if(${SOFA_FLOATING_POINT_TYPE} STREQUAL float)
    set(SOFA_WITH_FLOAT 1)
endif()
if(${SOFA_FLOATING_POINT_TYPE} STREQUAL double)
    set(SOFA_WITH_DOUBLE 1)
endif()
if(${SOFA_FLOATING_POINT_TYPE} STREQUAL both)
    set(SOFA_WITH_FLOAT 1)
    set(SOFA_WITH_DOUBLE 1)
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

## Packages
add_subdirectory(applications/packages)

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


# Install configuration
#install(FILES "${CMAKE_BINARY_DIR}/CMakeCache.txt" DESTINATION . COMPONENT headers)
install(FILES "${CMAKE_SOURCE_DIR}/README.md" DESTINATION . COMPONENT applications)
install(FILES "${CMAKE_SOURCE_DIR}/CHANGELOG.md" DESTINATION . COMPONENT applications)
install(FILES "${CMAKE_SOURCE_DIR}/LICENSE.LGPL.txt" DESTINATION . COMPONENT applications)
install(FILES "${CMAKE_SOURCE_DIR}/Authors.txt" DESTINATION . COMPONENT applications)

## Create etc/installedSofa.ini: it contains the paths to share/ and examples/. Compare to the Sofa.ini in the
## build directory, this one contains relative paths to the installed resource directory.
set(SHARE_DIR "../share/sofa")
set(EXAMPLES_DIR "../share/sofa/examples")
configure_file(${SOFA_KERNEL_SOURCE_DIR}/etc/sofa.ini.in "${CMAKE_BINARY_DIR}/etc/installedSofa.ini")
install(FILES "${CMAKE_BINARY_DIR}/etc/installedSofa.ini" DESTINATION etc RENAME sofa.ini COMPONENT applications)


option(SOFA_INSTALL_RESOURCES_FILES "Copy resources files (etc/, share/, examples/) when installing" ON)
## Install resource files
if(SOFA_INSTALL_RESOURCES_FILES)
    install(DIRECTORY share/ DESTINATION share/sofa COMPONENT resources)
    install(DIRECTORY examples/ DESTINATION share/sofa/examples COMPONENT resources)
endif()


#CPack install
SET(CPACK_PACKAGE_VERSION_MAJOR "17")
SET(CPACK_PACKAGE_VERSION_MINOR "12")
SET(CPACK_PACKAGE_VERSION_PATCH "00")
SET(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}")
set(CPACK_PACKAGE_NAME "SOFA v${CPACK_PACKAGE_VERSION}")
set(CPACK_PACKAGE_FILE_NAME "SOFA_v${CPACK_PACKAGE_VERSION}")
set(CPACK_PACKAGE_VENDOR "The SOFA Team")
#set(CPACK_PACKAGE_CONTACT "contact@sofa-framework.org")
set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_SOURCE_DIR}/README.md")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Real-time multi-physics simulation with an emphasis on medical simulation.")
set(CPACK_PACKAGE_EXECUTABLES "runSofa" "runSofa")

set(CPACK_COMPONENTS_ALL applications headers Eigen_headers GTest_headers libraries resources)

set(CPACK_COMPONENT_APPLICATIONS_DISPLAY_NAME "runSofa Application")
set(CPACK_COMPONENT_HEADERS_DISPLAY_NAME "C++ Headers")
set(CPACK_COMPONENT_EIGEN_HEADERS_DISPLAY_NAME "Eigen Headers")
set(CPACK_COMPONENT_GTEST_HEADERS_DISPLAY_NAME "GTest Headers")
set(CPACK_COMPONENT_LIBRARIES_DISPLAY_NAME "Libraries")
set(CPACK_COMPONENT_RESOURCES_DISPLAY_NAME "Resources")

set(CPACK_COMPONENT_APPLICATIONS_GROUP "Runtime")
set(CPACK_COMPONENT_HEADERS_GROUP "Development")
set(CPACK_COMPONENT_EIGEN_HEADERS_GROUP "Development")
set(CPACK_COMPONENT_GTEST_HEADERS_GROUP "Development")
set(CPACK_COMPONENT_LIBRARIES_GROUP "Development")
set(CPACK_COMPONENT_RESOURCES_GROUP "Runtime")

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE.LGPL.txt")
set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")
set(CPACK_RESOURCE_FILE_WELCOME "${CMAKE_SOURCE_DIR}/README.md")

if(WIN32)
    # There is a bug in NSI that does not handle full unix paths properly. Make
    # sure there is at least one set of four (4) backlasshes.
    set(CPACK_PACKAGE_INSTALL_DIRECTORY "SOFA_v${CPACK_PACKAGE_VERSION}")
    set(CPACK_PACKAGE_ICON "${CMAKE_SOURCE_DIR}/share/icons\\\\SOFA.png")
    set(CPACK_NSIS_INSTALLED_ICON_NAME "bin\\\\runSofa.exe")
    set(CPACK_NSIS_DISPLAY_NAME "SOFA v${CPACK_PACKAGE_VERSION}")
    set(CPACK_NSIS_HELP_LINK "https:\\\\\\\\www.sofa-framework.org")
    set(CPACK_NSIS_URL_INFO_ABOUT "https:\\\\\\\\www.sofa-framework.org")
    set(CPACK_NSIS_CONTACT "contact@sofa-framework.org")
    set(CPACK_NSIS_INSTALL_ROOT "$PROFILE")
elseif(APPLE)
    #set(CPACK_PACKAGE_INSTALL_DIRECTORY "SOFA/v${CPACK_PACKAGE_VERSION}")

    #set(CPACK_DMG_BACKGROUND_IMAGE "${CMAKE_SOURCE_DIR}/back.png")
else()
    set(CPACK_PACKAGE_INSTALL_DIRECTORY "SOFA/v${CPACK_PACKAGE_VERSION}")

    find_package(ICU COMPONENTS data i18n uc)
    if(ICU_FOUND)
        sofa_install_libraries(${ICU_LIBRARIES})
    endif()
endif()

#dirty hack to pack component we want (named BundlePack from runSofa for example, and that will install .app + share)
#if not set, it will install everything as usual
#TODO: Redesign of a better architecture about bundling (runSofa ? libraries ? modeler ? other application ?)
get_property(TEMP_CPACK_COMPONENTS_ALL GLOBAL PROPERTY RUNSOFA_CPACK_COMPONENTS_ALL )
if(TEMP_CPACK_COMPONENTS_ALL)
    set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
    set(CPACK_COMPONENTS_ALL ${TEMP_CPACK_COMPONENTS_ALL})
endif()

if(NOT ${CPACK_PACKAGE_VERSION_PATCH} STREQUAL "dev" AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/package.cmake")
    message("Using package.cmake")
    include( "package.cmake" )
endif()

INCLUDE(CPack)
