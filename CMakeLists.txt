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

sofa_resetInternalCacheVariables()

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
    sofa_printConfigurationReport()
endif()
