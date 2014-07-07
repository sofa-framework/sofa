cmake_minimum_required(VERSION 2.8.8)

# automatically use ccache if it is installed
find_program(ccache_cmd NAMES ccache)
if(ccache_cmd)
    SET_PROPERTY(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    SET_PROPERTY(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
    message(STATUS "Using ccache for building")
endif()


set(SOLUTION_NAME "Sofa")
project(${SOLUTION_NAME})

# Enable the organisation in folders for generators that support
# it. (E.g. some versions of Visual Studio have 'solution folders')
set_property(GLOBAL
PROPERTY USE_FOLDERS ON)

include(cmake/utils.cmake)       # utils macros
include(cmake/environment.cmake) # paths
include(cmake/functions.cmake)   # sofa build functions
include(cmake/options.cmake)     # user options


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
    configure_file("${SOFA_FRAMEWORK_DIR}/sofa/config.h.in"
                   "${SOFA_BUILD_DIR}/misc/include/sofa/config.h")

    sofa_save_option_list("misc/options.txt")
    sofa_save_compiler_definitions("misc/compiler-definitions.txt")
    sofa_save_dependencies("misc/dependencies.txt")
    sofa_save_complete_dependencies("misc/full-dependencies.txt")
    sofa_print_configuration_report()

    include(cmake/doxygen.cmake)
endif()
