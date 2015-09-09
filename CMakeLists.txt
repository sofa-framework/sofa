cmake_minimum_required(VERSION 2.8.8)

set(SOLUTION_NAME "Sofa")
project(${SOLUTION_NAME})

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
set_property(GLOBAL
PROPERTY USE_FOLDERS ON)

include(cmake/utils.cmake)       # utils macros
include(cmake/environment.cmake) # paths
include(cmake/functions.cmake)   # sofa build functions
include(cmake/options.cmake)     # user options

if(NOT PRECONFIGURE_DONE)
    # Load pre-checked options
    if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/cmake/custom_options.cmake" )
        message("Set Custom options")
        include(cmake/custom_options.cmake)
    endif()

    message("")
    message(">>> SOFA framework pre-configuration complete")
    message(">>> Select your options and launch 'configure' or re-run cmake")
    message("")
    set(PRECONFIGURE_DONE 1 CACHE INTERNAL "")

else()
    include(cmake/externals.cmake)
    include(cmake/buildFlags.cmake)
    include(cmake/dependencies.cmake)

    # Generate sofa/version.h
    set(SOFA_VERSION "150900")
    set(SOFA_VERSION_STR "\"15.09\"")
    configure_file("${SOFA_FRAMEWORK_DIR}/sofa/version.h.in"
                   "${SOFA_BUILD_DIR}/misc/include/sofa/version.h")

    # Generate config.h header
    configure_file("${SOFA_FRAMEWORK_DIR}/sofa/config.h.in"
                   "${SOFA_BUILD_DIR}/misc/include/sofa/config.h")

    # Configuration file
    set(SHARE_DIR ${SOFA_SRC_DIR}/share)
    set(EXAMPLES_DIR ${SOFA_SRC_DIR}/examples)
    configure_file(etc/sofa.ini.in ${CMAKE_BINARY_DIR}/etc/sofa.ini)

    sofa_save_option_list("misc/options.txt")
    sofa_save_compiler_definitions("misc/compiler-definitions.txt")
    sofa_save_dependencies("misc/dependencies.txt")
    sofa_save_complete_dependencies("misc/full-dependencies.txt")
    sofa_print_configuration_report()

    include(cmake/doxygen.cmake)
endif()
