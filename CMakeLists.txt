cmake_minimum_required(VERSION 2.8.8)

set(SOLUTION_NAME "Sofa")
project(${SOLUTION_NAME})

# MSVC: Use solution folders.
if(MSVC)
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif()

include(cmake/environment.cmake)
include(cmake/functions.cmake)
include(cmake/options.cmake)

if(NOT PRECONFIGURE_DONE)
    message("")
    message(">>> SOFA framework pre-configuration complete")
    message(">>> Select your options and launch 'configure' or re-run cmake")
    message("")
    set(PRECONFIGURE_DONE 1 CACHE INTERNAL "")
else()
    include(cmake/externals.cmake)
    include(cmake/buildFlags.cmake)
    include(cmake/dependencies.cmake)

    # Generate config.h header
    configure_file("${SOFA_FRAMEWORK_DIR}/sofa/config.h.in" "${SOFA_BUILD_DIR}/misc/include/sofa/config.h")

    sofa_save_option_list("misc/options.txt")
    sofa_save_compiler_definitions("misc/compiler-definitions.txt")
    sofa_save_dependencies("misc/dependencies.txt")
    sofa_save_complete_dependencies("misc/full-dependencies.txt")
    sofa_print_configuration_report()
endif()
