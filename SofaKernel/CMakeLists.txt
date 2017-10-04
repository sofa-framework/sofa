cmake_minimum_required(VERSION 3.1)
project(SofaKernel)

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

## Set the output directories globally
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

## Environment
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaFramework")
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR}/extlibs)
# list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR}/applications/plugins)
# list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR}/applications/projects)

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

        if(CMAKE_COMPILER_IS_GNUCXX)

            EXEC_PROGRAM(${CMAKE_CXX_COMPILER} ARGS ${CMAKE_CXX_COMPILER_ARG1} -dumpversion OUTPUT_VARIABLE GCC_VERSION)

            if(GCC_VERSION VERSION_GREATER 4.7 OR GCC_VERSION VERSION_EQUAL 4.7)
                set(CXX11_FLAG "-std=c++11" CACHE STRING "c++11 compiler flag")
            elseif(GCC_VERSION VERSION_GREATER 4.3 OR GCC_VERSION VERSION_EQUAL 4.3)
                message(WARNING "c++0x activated. g++ compiler with a version >= 4.7 is needed to fully support c++11.")
                set(CXX11_FLAG "-std=c++0x" CACHE STRING "c++11 compiler flag")
            else ()
                message("Warning: a g++ compiler with a version >= 4.3 is needed to support c++11.")
            endif()

        else() # not gcc -> clang?
            # TODO detect clang versions ?
            set(CXX11_FLAG "-std=c++11" CACHE STRING "c++11 compiler flag")
        endif()

    endif(SOFA_USE_C++11)

endif(NOT WIN32)


## Try to find an installed Boost (1.54.0 or higher).
set(BOOST_MIN_VERSION "1.54.0")
find_package(Boost ${BOOST_MIN_VERSION} REQUIRED)

### Dependency pack for Windows
if(MSVC)

    #define BOOST_ALL_DYN_LINK needed for dynamic linking with boost libraries
    add_definitions(-DBOOST_ALL_DYN_LINK)

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
            file(COPY ${DLLS} DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CONFIG}")
        endforeach()
    else()                      # Single-config generator (nmake)
        file(COPY ${DLLS} DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
    endif()
    ## Install DLLs
    install(FILES ${DLLS} DESTINATION bin)
    install(FILES ${LIBS} DESTINATION lib)
    install(DIRECTORY ${DEPENDENCY_PACK_DIR}/include/ DESTINATION include)

endif()

# Create etc/sofa.ini: it contains the paths to share/ and examples/. In the
# build directory, it points to the source tree, whereas in the install
# directory, it contains to relative paths to the installed resource directory.
configure_file(etc/sofa.ini.in "${CMAKE_BINARY_DIR}/etc/sofa.ini")
configure_file(etc/sofa.ini.in "${CMAKE_BINARY_DIR}/etc/installedSofa.ini")
install(FILES "${CMAKE_BINARY_DIR}/etc/installedSofa.ini" DESTINATION etc RENAME sofa.ini)

### Extlibs
option(SOFA_USE_MASK "Use mask optimization" ON)
if(SOFA_USE_MASK)
    add_definitions(-DSOFA_USE_MASK)
endif()

add_subdirectory(extlibs)

### Main Sofa subdirectories

add_subdirectory(SofaFramework)
add_subdirectory(SofaSimulation)
add_subdirectory(SofaBase)
add_subdirectory(SofaCommon)


## Custom
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/custom.cmake")
    message("Adding custom file")
    include( "custom.cmake" )
endif()


## when installing, keep an eye on options/params/sources used for compilation
## this should be internal and not delivered, but this is definitively useful
sofa_install_git_version( "sofa" ${CMAKE_CURRENT_SOURCE_DIR} )
install(FILES "${CMAKE_BINARY_DIR}/CMakeCache.txt" DESTINATION .)

#CPack install
SET(CPACK_PACKAGE_VERSION "17.dev.0")
SET(CPACK_PACKAGE_VERSION_MAJOR "17")
SET(CPACK_PACKAGE_VERSION_MINOR "dev")
SET(CPACK_PACKAGE_VERSION_PATCH "0")
SET(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENCE.txt")
SET(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/readme.txt")

