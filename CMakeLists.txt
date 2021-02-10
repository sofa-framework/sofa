cmake_minimum_required(VERSION 3.12)
project(Sofa) # Cannot use VERSION with patch like "00"

# Manually define VERSION
set(Sofa_VERSION_MAJOR 20)
set(Sofa_VERSION_MINOR 12)
set(Sofa_VERSION_PATCH 99)
set(Sofa_VERSION ${Sofa_VERSION_MAJOR}.${Sofa_VERSION_MINOR}.${Sofa_VERSION_PATCH})

set(SOFA_KERNEL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/SofaKernel" CACHE STRING "Path to SofaKernel")
if(NOT EXISTS ${SOFA_KERNEL_SOURCE_DIR})
    message(WARNING "SofaKernel is not present or is inaccessible. Check if ${SOFA_KERNEL_SOURCE_DIR} exists or is accessible. Backing up to default value.")
    set(SOFA_KERNEL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/SofaKernel" CACHE STRING "Path to SofaKernel" FORCE)
endif()

## Default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

include(CMakeDependentOption)

# Enable the organisation in folders for generators that support
# it. (E.g. some versions of Visual Studio have 'solution folders')
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Option for packaging
option(SOFA_BUILD_RELEASE_PACKAGE "Run package specific configure" OFF)

# Option to accelerate the compilation
# see https://cmake.org/cmake/help/v3.16/command/target_precompile_headers.html
# and https://cmake.org/cmake/help/v3.16/prop_tgt/DISABLE_PRECOMPILE_HEADERS.html
cmake_dependent_option(SOFA_BUILD_WITH_PCH_ENABLED 
    "Compile SOFA using precompiled header (to accelerate the build process)" OFF
    "CMAKE_VERSION VERSION_GREATER_EQUAL 3.16" OFF)
if(SOFA_BUILD_WITH_PCH_ENABLED)
    message("-- Precompiled headers: enabled (SOFA_BUILD_WITH_PCH_ENABLED is ON).")
else()
    message("-- Precompiled headers: disabled (SOFA_BUILD_WITH_PCH_ENABLED is OFF or CMake < 3.16).")
    set(DISABLE_PRECOMPILE_HEADERS ON)
endif()

## Change default install prefix
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install CACHE PATH "Install path prefix, prepended onto install directories." FORCE)
endif()
if(APPLE AND SOFA_BUILD_APP_BUNDLE)
    set(SOFA_BUILD_RELEASE_PACKAGE ON)
    set(CMAKE_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}/runSofa.app/Contents/MacOS)
    message("SOFA_BUILD_APP_BUNDLE = ${SOFA_BUILD_APP_BUNDLE}\n"
        "  Forcing SOFA_BUILD_RELEASE_PACKAGE = ${SOFA_BUILD_RELEASE_PACKAGE}\n"
        "  Forcing CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}"
        )
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
list(APPEND CMAKE_MODULE_PATH "${SOFA_KERNEL_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${SOFA_KERNEL_SOURCE_DIR}/cmake/Modules")
list(APPEND CMAKE_MODULE_PATH "${SOFA_KERNEL_SOURCE_DIR}/SofaFramework")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_BINARY_DIR}/cmake")
list(APPEND CMAKE_IGNORE_PATH "${CMAKE_INSTALL_PREFIX}") # ignore install directory for findXXX commands

install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/FindEigen3.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/macdeployqt.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/windeployqt.cmake"
    DESTINATION lib/cmake/Modules
    COMPONENT headers
    )

include(SofaMacros)

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
    set(CMAKE_INSTALL_RPATH
        "$ORIGIN/../lib"
        "$$ORIGIN/../lib"
        )

    if(APPLE)
        set(CMAKE_MACOSX_RPATH ON)
        list(APPEND CMAKE_INSTALL_RPATH
            "@loader_path/../lib"
            "@executable_path/../lib"
            "@loader_path/../../Frameworks"
            "@executable_path/../../Frameworks"
            )
        set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    endif()
endif(UNIX)

include(CompilerOptions)

## Boost (1.54.0 or higher) is now mandatory.
set(BOOST_MIN_VERSION "1.54.0")
set(Boost_NO_BOOST_CMAKE TRUE)
find_package(Boost ${BOOST_MIN_VERSION} QUIET)
if(NOT Boost_FOUND)
    if(WIN32)
        message(FATAL_ERROR "Boost (${BOOST_MIN_VERSION} or higher) is mandatory.\n"
            "Note that Boost autodetection on Windows was removed.\n"
            "Please follow CMake documentation to find Boost on your system.\n"
            "Hint: set BOOST_ROOT to your Boost location."
            )
    else()
        message(FATAL_ERROR "Boost (${BOOST_MIN_VERSION} or higher) is mandatory.")
    endif()
endif()

### Windows config
if(MSVC)
    # WinDepPack
    set(SOFA_DEPENDENCY_PACK_DIR "${CMAKE_SOURCE_DIR}" CACHE PATH "Directory containing Windows Dependency Pack")
    if(NOT EXISTS ${SOFA_DEPENDENCY_PACK_DIR})
        # force back to default value
        set(SOFA_DEPENDENCY_PACK_DIR "${CMAKE_SOURCE_DIR}" CACHE PATH "Directory containing Windows Dependency Pack" FORCE)
    endif()
    list(APPEND CMAKE_INCLUDE_PATH ${SOFA_DEPENDENCY_PACK_DIR}/include)
    if(CMAKE_CL_64)
        list(APPEND CMAKE_LIBRARY_PATH ${SOFA_DEPENDENCY_PACK_DIR}/lib/win64)
    else()
        list(APPEND CMAKE_LIBRARY_PATH ${SOFA_DEPENDENCY_PACK_DIR}/lib/win32)
    endif()
    install(DIRECTORY ${SOFA_DEPENDENCY_PACK_DIR}/include/ DESTINATION include/extlibs/WinDepPack COMPONENT headers)
    install(DIRECTORY ${SOFA_DEPENDENCY_PACK_DIR}/licenses/ DESTINATION licenses COMPONENT applications)
endif()

### Mask
option(SOFA_USE_MASK "Use mask optimization" OFF)

### SOFA_DEV_TOOL
option(SOFA_WITH_DEVTOOLS "Compile with developement extra features." ON)

### Threading
option(SOFA_WITH_THREADING "Compile sofa with thread-safetiness support (PARTIAL/EXPERIMENTAL)" ON)

### Testing
option(SOFA_BUILD_TESTS "Compile the automatic tests for Sofa, along with the gtest library." ON)
if(SOFA_BUILD_TESTS OR SOFA_BUILD_RELEASE_PACKAGE)
    # Enable testing features of cmake, like the add_test() command.
    enable_testing()
    add_subdirectory(extlibs/gtest)
endif()

## Active or not the use of ccache
option(SOFA_USE_CCACHE "Compile using ccache optimization" OFF)
if(SOFA_USE_CCACHE)
    find_program(CCACHE_PROGRAM ccache)
    if(CCACHE_PROGRAM)
        # Support Unix Makefiles and Ninja
        #TODO replace by <LANG>_COMPILER_LAUNCHER when min cmake version > 3.4
        set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
        set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CCACHE_PROGRAM}")
    else()
        message(WARNING "ccache not found, disabling option")
        set(SOFA_USE_CCACHE OFF CACHE bool "Compile using ccache optimization" FORCE)
    endif()
endif()

### Ninja build pools
option(SOFA_NINJA_BUILD_POOLS "Activate the Ninja build pools feature, to limit the cores used by specific targets" OFF)

# This option is useful to compile SOFA + internal plugins without SofaPython soft dependencies (SofaExporter, SofaTest, Compliant, image, ...)
# it simplifies binaries generation and future cohabitation between SofaPython and SofaPython3
option(SOFA_ENABLE_SOFT_DEPS_TO_SOFAPYTHON "Enable all soft dependencies to SofaPython" ON)

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

### Deprecated components
option(SOFA_WITH_DEPRECATED_COMPONENTS "Compile SOFA with all deprecated components" ON)
if(SOFA_WITH_DEPRECATED_COMPONENTS)
    message(DEPRECATION "Deprecated components are activated (SOFA_WITH_DEPRECATED_COMPONENTS variable is ON)")
endif()

### Extlibs
add_subdirectory(extlibs)

### Main Sofa subdirectories
add_subdirectory(${SOFA_KERNEL_SOURCE_DIR}/SofaFramework)
add_subdirectory(${SOFA_KERNEL_SOURCE_DIR}/SofaSimulation)
if(SOFA_BUILD_TESTS OR SOFA_BUILD_RELEASE_PACKAGE)
    # Depends on SofaSimulation
    add_subdirectory(tools/SofaGTestMain)
endif()

add_subdirectory(${SOFA_KERNEL_SOURCE_DIR}/SofaBase)

## Modules
## in the consortium convention a module is a similar to a plugin the difference is that
## a module is hosted and managed by the consortium while plugins are hosted and managed by third parties.
add_subdirectory(${SOFA_KERNEL_SOURCE_DIR}/modules)
add_subdirectory(modules)

# This is all the applications GUI stuff. As there is a dependency to this one
# In scene creator and SofaPython...it is located here.
add_subdirectory(SofaGui)

if(SOFA_ENABLE_SOFT_DEPS_TO_SOFAPYTHON)
    # SofaTest depend on SofaPython, so we add SofaPython before.
    # and SofaPython needs SofaGui...
    sofa_add_plugin(applications/plugins/SofaPython SofaPython)
    sofa_add_module(modules/SofaExporter SofaExporter ON)
endif()

## Collections
## A collection is a "meta module" and containing several modules (usable at find level [find_package] and include/link level [add_library])
# SofaAdvanced is legacy and deprecated since 20.06
set(SOFA_LIST_OF_DEPRECATED_COLLECTIONS ${SOFA_KERNEL_SOURCE_DIR}/SofaCommon SofaGeneral SofaAdvanced SofaMisc)
option(SOFA_BUILD_DEPRECATED_COLLECTIONS "Build legacy collections (${SOFA_LIST_OF_DEPRECATED_COLLECTIONS})." ON)
if(SOFA_BUILD_DEPRECATED_COLLECTIONS)
    foreach(deprecated_collection ${SOFA_LIST_OF_DEPRECATED_COLLECTIONS})
        add_subdirectory(${deprecated_collection})
    endforeach()
endif()
## Customized collections
# SofaComponentAll is a convenient collection to get all Sofa features
add_subdirectory(applications/collections)

## Scene creator option
option(SOFA_BUILD_SCENECREATOR "Compile the C++ API located in applications/plugins/SceneCreator" OFF)

## Tutorials option
option(SOFA_BUILD_TUTORIALS "Build (most of) the tutorials available." OFF)

# Activate SceneCreator when required
if(SOFA_BUILD_TESTS OR SOFA_BUILD_TUTORIALS OR SOFA_BUILD_RELEASE_PACKAGE)
    set(SOFA_BUILD_SCENECREATOR ON CACHE BOOL "" FORCE)
endif()


# Library used by SOFA_BUILD_TESTS and SOFA_BUILD_TUTORIALS
if(SOFA_BUILD_SCENECREATOR)
    add_subdirectory(applications/plugins/SceneCreator)
endif()

# Tests : define subdirectories
if(SOFA_BUILD_TESTS OR SOFA_BUILD_RELEASE_PACKAGE)
    # Library used to write high level tests involving many components.
    add_subdirectory(applications/plugins/SofaTest)
endif()

## Plugins
add_subdirectory(applications/plugins)

if(SOFA_BUILD_TESTS)
    # Tests for all the modules, written using SofaTest.
    add_subdirectory(tests)
endif()

if(SOFA_BUILD_TESTS AND PLUGIN_SOFAPYTHON AND SOFA_ENABLE_SOFT_DEPS_TO_SOFAPYTHON)
    # SofaPython_test is a special case, because it depends on
    # SofaTest, which may depend on SofaPython...
    add_subdirectory(applications/plugins/SofaPython/SofaPython_test)
endif()

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

option(SOFA_INSTALL_RESOURCES_FILES "Copy resources files (etc/, share/, examples/) when installing" ON)
## Install resource files
if(SOFA_INSTALL_RESOURCES_FILES)
    install(DIRECTORY share/ DESTINATION share/sofa COMPONENT resources)
    install(DIRECTORY examples/ DESTINATION share/sofa/examples COMPONENT resources)
endif()

file(WRITE "${CMAKE_BINARY_DIR}/plugins/README.txt" "This folder will be automatically scanned by the Plugin Manager.")
install(DIRECTORY ${CMAKE_BINARY_DIR}/plugins/ DESTINATION plugins COMPONENT resources)

# Post-install scripts (must be the last add_subdirectory)
add_subdirectory(tools/postinstall-fixup)

#######################
# CPack configuration #
#######################

################
# Package config
include(CPackComponent)
if(WIN32 AND ${CMAKE_VERSION} VERSION_LESS "3.13.0")
    include(CPackNSIS)
endif()
include(CPackIFW)
set(CPACK_PACKAGE_VERSION "${Sofa_VERSION}")
set(CPACK_PACKAGE_NAME "SOFA v${CPACK_PACKAGE_VERSION}")
set(CPACK_PACKAGE_VENDOR "The SOFA Team")
set(CPACK_PACKAGE_CONTACT "contact@sofa-framework.org")
set(CPACK_PACKAGE_DESCRIPTION "Real-time multi-physics simulation with an emphasis on medical simulation.")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Real-time multi-physics simulation with an emphasis on medical simulation.")
set(CPACK_PACKAGE_EXECUTABLES "runSofa" "runSofa")
set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_SOURCE_DIR}/README.md")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://www.sofa-framework.org")
set(CPACK_PACKAGE_FILE_NAME "SOFA_v${CPACK_PACKAGE_VERSION}")
if(WIN32)
    set(CPACK_PACKAGE_ICON "${CMAKE_SOURCE_DIR}/share/icons\\\\SOFA.png")
    set(CPACK_PACKAGE_INSTALL_DIRECTORY "SOFA\\\\v${CPACK_PACKAGE_VERSION}")
    if(CMAKE_EXE_LINKER_FLAGS MATCHES ".*machine:x64")
        set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_FILE_NAME}_Win64")
    else() # x86
        set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_FILE_NAME}_Win32")
    endif()
elseif(UNIX)
    set(CPACK_PACKAGE_ICON "${CMAKE_SOURCE_DIR}/share/icons/SOFA.png")
    set(CPACK_PACKAGE_INSTALL_DIRECTORY "SOFA/v${CPACK_PACKAGE_VERSION}")
    if(APPLE)
        set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_FILE_NAME}_MacOS")
    else()
        set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_FILE_NAME}_Linux")
    endif()
endif()
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE.LGPL.txt")
set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")
set(CPACK_RESOURCE_FILE_WELCOME "${CMAKE_SOURCE_DIR}/README.md")
################

##################
# CPack components
cpack_add_component_group(runtime
    DISPLAY_NAME "Runtime"
    DESCRIPTION "Components needed to run a simulation with ${CPACK_PACKAGE_NAME}"
    EXPANDED
    )
cpack_add_component_group(development
    DISPLAY_NAME "Development"
    DESCRIPTION "Components needed to write code based on ${CPACK_PACKAGE_NAME}"
    EXPANDED
    )

cpack_add_component(applications
    DISPLAY_NAME "runSofa Application"
    GROUP runtime
    )
cpack_add_component(headers
    DISPLAY_NAME "C++ Headers"
    GROUP development
    )
cpack_add_component(GTest_headers
    DISPLAY_NAME "GTest Headers"
    GROUP development
    )
cpack_add_component(libraries
    DISPLAY_NAME "Libraries"
    GROUP development
    )
cpack_add_component(resources
    DISPLAY_NAME "Resources"
    GROUP runtime
    )

set(CPACK_COMPONENTS_ALL applications headers GTest_headers libraries resources)

set(CPACK_COMPONENT_APPLICATIONS_DISPLAY_NAME "runSofa Application")
set(CPACK_COMPONENT_HEADERS_DISPLAY_NAME "C++ Headers")
set(CPACK_COMPONENT_GTEST_HEADERS_DISPLAY_NAME "GTest Headers")
set(CPACK_COMPONENT_LIBRARIES_DISPLAY_NAME "Libraries")
set(CPACK_COMPONENT_RESOURCES_DISPLAY_NAME "Resources")

set(CPACK_COMPONENT_APPLICATIONS_GROUP "Runtime")
set(CPACK_COMPONENT_HEADERS_GROUP "Development")
set(CPACK_COMPONENT_GTEST_HEADERS_GROUP "Development")
set(CPACK_COMPONENT_LIBRARIES_GROUP "Development")
set(CPACK_COMPONENT_RESOURCES_GROUP "Runtime")
##################

######################
# IFW Generator config
if(CPACK_BINARY_IFW)
    set(CPACK_IFW_PACKAGE_NAME "${CPACK_PACKAGE_NAME}")
    set(CPACK_IFW_PACKAGE_TITLE "${CPACK_PACKAGE_NAME}")
    set(CPACK_IFW_PRODUCT_URL "${CPACK_PACKAGE_HOMEPAGE_URL}")
    set(CPACK_IFW_PACKAGE_LOGO "${CPACK_PACKAGE_ICON}")
    set(CPACK_IFW_TARGET_DIRECTORY "@HomeDir@/${CPACK_PACKAGE_INSTALL_DIRECTORY}")
    if(WIN32)
         set(CPACK_IFW_PACKAGE_ICON "${CMAKE_SOURCE_DIR}/applications/projects/runSofa/runSofa.ico")
    elseif(APPLE)
         set(CPACK_IFW_PACKAGE_ICON "${CMAKE_SOURCE_DIR}/applications/projects/runSofa/runSofa.icns")
    endif()

    cpack_ifw_configure_component_group(runtime
        SORTING_PRIORITY 50
        DEFAULT TRUE
        EXPANDED
        SCRIPT "${CMAKE_SOURCE_DIR}/scripts/qtifw/install.qs"
        )
    cpack_ifw_configure_component_group(development
        SORTING_PRIORITY 10
        DEFAULT TRUE
        EXPANDED
        )

    cpack_ifw_configure_component(applications
        DISPLAY_NAME "runSofa Application"
        DEPENDS runtime
        )
    cpack_ifw_configure_component(headers
        DISPLAY_NAME "C++ Headers"
        DEPENDS development
        )
    cpack_ifw_configure_component(GTest_headers
        DISPLAY_NAME "GTest Headers"
        DEPENDS development
        )
    cpack_ifw_configure_component(libraries
        DISPLAY_NAME "Libraries"
        DEPENDS development
        )
    cpack_ifw_configure_component(resources
        DISPLAY_NAME "Resources"
        DEPENDS runtime
        )
endif()
######################

#######################
# NSIS Generator config
if(CPACK_BINARY_NSIS)
    # There is a bug in NSIS that does not handle full unix paths properly. Make
    # sure there is at least one set of four (4) backslashes.
    set(CPACK_NSIS_MUI_ICON "${CMAKE_SOURCE_DIR}/applications/projects/runSofa/runSofa.ico")
    set(CPACK_NSIS_INSTALLED_ICON_NAME "bin\\\\runSofa.exe")
    set(CPACK_NSIS_DISPLAY_NAME ${CPACK_PACKAGE_NAME})
    set(CPACK_NSIS_PACKAGE_NAME ${CPACK_PACKAGE_NAME})
    set(CPACK_NSIS_HELP_LINK ${CPACK_PACKAGE_HOMEPAGE_URL})
    set(CPACK_NSIS_URL_INFO_ABOUT ${CPACK_PACKAGE_HOMEPAGE_URL})
    set(CPACK_NSIS_MODIFY_PATH ON)
    set(CPACK_NSIS_CONTACT "contact@sofa-framework.org")
    set(CPACK_NSIS_INSTALL_ROOT "$PROFILE")
    string(CONCAT EXTRA_INSTALL_COMMAND "ExecShell \\\"open\\\" \\\"https://www.sofa-framework.org/thank-you?sofa=" ${CPACK_PACKAGE_VERSION} "&os=windows\\\"")
    set(CPACK_NSIS_EXTRA_INSTALL_COMMANDS "${EXTRA_INSTALL_COMMAND}")
    set(CPACK_NSIS_MENU_LINKS
        "https://www.sofa-framework.org/documentation/" "SOFA User Documentation"
        "https://www.sofa-framework.org/api/" "SOFA Developper API"
        "https://www.sofa-framework.org/community/forum/" "SOFA Forum")
endif()
#######################

if(APPLE AND SOFA_BUILD_APP_BUNDLE)
    set(CPACK_BINARY_DRAGNDROP ON)
    set(CPACK_GENERATOR DragNDrop)

    # Monolithic install containing everything
    unset(CPACK_COMPONENTS_ALL)
    set(CPACK_MONOLITHIC_INSTALL ON)

    # Force CPack install dir to $ENV{DESTDIR}/runSofa.app/Contents/MacOS
    # CPack always adds "$ENV{DESTDIR}" before CPACK_INSTALL_PREFIX
    set(CPACK_SET_DESTDIR TRUE)
    set(CPACK_INSTALL_PREFIX "/runSofa.app/Contents/MacOS")
endif()

include(CPack)
