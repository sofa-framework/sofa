cmake_minimum_required(VERSION 3.12)
project(SofaNG)

include(CMakeDependentOption)

# Manually define VERSION
set(Sofa_VERSION_MAJOR 20)
set(Sofa_VERSION_MINOR 12)
set(Sofa_VERSION_PATCH 99)
set(Sofa_VERSION ${Sofa_VERSION_MAJOR}.${Sofa_VERSION_MINOR}.${Sofa_VERSION_PATCH})

set(SOFA_VERSION_STR "\"${Sofa_VERSION}\"")
set(SOFA_VERSION "${Sofa_VERSION_MAJOR}${Sofa_VERSION_MINOR}${Sofa_VERSION_PATCH}")

set(SOFA_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." CACHE PATH "Path to Sofa")
set(SOFA_KERNEL_SOURCE_DIR "${SOFA_SOURCE_DIR}/SofaKernel" CACHE PATH "Path to SofaKernel")

if(NOT EXISTS ${SOFA_KERNEL_SOURCE_DIR}/modules/Sofa.Config)
    message(FATAL_ERROR "Cannot find a correct path for SofaKernel.")
endif()

## Default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# Output Directories definition
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

# Remove generated CMake files, this prevents CMake from finding packages that
# were disabled (like, unchecked in cmake-gui) after being built once.
file(REMOVE_RECURSE ${CMAKE_BINARY_DIR}/cmake)
file(REMOVE_RECURSE ${CMAKE_BINARY_DIR}/lib/cmake)

# Sofa.Config sets the environment (options, compiler flags, global variables)
add_subdirectory(${SOFA_KERNEL_SOURCE_DIR}/modules/Sofa.Config ${CMAKE_CURRENT_BINARY_DIR}/Sofa.Config)

## Custom Environment
list(APPEND CMAKE_MODULE_PATH "${SOFA_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${SOFA_SOURCE_DIR}/cmake/Modules")
list(APPEND CMAKE_IGNORE_PATH "${CMAKE_INSTALL_PREFIX}") # ignore install directory for findXXX commands

install(FILES
    "${SOFA_SOURCE_DIR}/cmake/Modules/FindEigen3.cmake"
    "${SOFA_SOURCE_DIR}/cmake/Modules/macdeployqt.cmake"
    "${SOFA_SOURCE_DIR}/cmake/Modules/windeployqt.cmake"
    DESTINATION lib/cmake/Modules
    COMPONENT headers
    )

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

### Windows config
if(MSVC)
    # WinDepPack
    set(SOFA_DEPENDENCY_PACK_DIR "${SOFA_SOURCE_DIR}" CACHE PATH "Directory containing Windows Dependency Pack")
    if(NOT EXISTS ${SOFA_DEPENDENCY_PACK_DIR})
        # force back to default value
        set(SOFA_DEPENDENCY_PACK_DIR "${SOFA_SOURCE_DIR}" CACHE PATH "Directory containing Windows Dependency Pack" FORCE)
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


### Testing
# option(SOFA_BUILD_TESTS "Compile the automatic tests for Sofa, along with the gtest library." ON)

### Extlibs
add_subdirectory(${SOFA_SOURCE_DIR}/extlibs  ${CMAKE_CURRENT_BINARY_DIR}/extlibs)

### Main Sofa subdirectories
option(SOFA_ENABLE_LEGACY_HEADERS "Enable the compatibility layer (headers located in legacy folders)." ON)
if(SOFA_ENABLE_LEGACY_HEADERS)
    message("Using legacy headers is enabled.")
    message("Headers from v20.12 are usable, but it is advised to change your code to use the new headers introduced in v21.06.")

    add_subdirectory(${SOFA_KERNEL_SOURCE_DIR}/modules/Sofa.Compat ${CMAKE_CURRENT_BINARY_DIR}/Sofa/Sofa.Compat)
endif()

### Main Sofa subdirectories
add_subdirectory(${SOFA_KERNEL_SOURCE_DIR}/SofaFramework ${CMAKE_CURRENT_BINARY_DIR}/Sofa/SofaFramework)
add_subdirectory(${SOFA_KERNEL_SOURCE_DIR}/SofaSimulation ${CMAKE_CURRENT_BINARY_DIR}/Sofa/SofaSimulation)

# if(SOFA_BUILD_TESTS)
#     add_subdirectory(${SOFA_KERNEL_SOURCE_DIR}/modules/Sofa.Testing ${CMAKE_CURRENT_BINARY_DIR}/Sofa/Sofa.Testing)
# endif()

if(SOFA_WITH_OPENGL)
    add_subdirectory(${SOFA_KERNEL_SOURCE_DIR}/modules/Sofa.GL ${CMAKE_CURRENT_BINARY_DIR}/Sofa/Sofa.GL)
endif()

# Components
add_subdirectory(Component ${CMAKE_CURRENT_BINARY_DIR}/Component)
# Applications
# add_subdirectory(Applications ${CMAKE_CURRENT_BINARY_DIR}/Applications)

# ## Scene creator option
# option(SOFA_BUILD_SCENECREATOR "Compile the C++ API located in applications/plugins/SceneCreator" OFF)

# ## Tutorials option
# option(SOFA_BUILD_TUTORIALS "Build (most of) the tutorials available." OFF)

# # Activate SceneCreator when required
# if(SOFA_BUILD_TESTS OR SOFA_BUILD_TUTORIALS OR SOFA_BUILD_RELEASE_PACKAGE)
#     set(SOFA_BUILD_SCENECREATOR ON CACHE BOOL "" FORCE)
# endif()


# # Library used by SOFA_BUILD_TESTS and SOFA_BUILD_TUTORIALS
# if(SOFA_BUILD_SCENECREATOR)
#     add_subdirectory(applications/plugins/SceneCreator)
# endif()

# # Tests : define subdirectories
# if(SOFA_BUILD_TESTS OR SOFA_BUILD_RELEASE_PACKAGE)
#     # Library used to write high level tests involving many components.
#     add_subdirectory(applications/plugins/SofaTest)
# endif()

# ## Plugins
# add_subdirectory(applications/plugins)

# ## Applications
# add_subdirectory(applications/projects)

## Build external projects at the same time
set(SOFA_EXTERNAL_DIRECTORIES "" CACHE STRING "list of paths separated by ';'")
if(NOT "${SOFA_EXTERNAL_DIRECTORIES}" STREQUAL "")
    foreach(dir ${SOFA_EXTERNAL_DIRECTORIES})
        get_filename_component(name ${dir} NAME) # Get the name of the actual directory
        message("Adding external directory: ${name} (${dir})")
        add_subdirectory(${dir} "${CMAKE_CURRENT_BINARY_DIR}/external_directories/${name}")
    endforeach()
endif()
