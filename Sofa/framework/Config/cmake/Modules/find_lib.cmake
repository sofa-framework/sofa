if(_FIND_LIB_CMAKE_INCLUDED_)
    return()
endif()
set(_FIND_LIB_CMAKE_INCLUDED_ true)


## CMAKE_DOCUMENTATION_START find_lib
##
##\\code
##FIND_LIB( MY_LIBRARY_VARIABLE MY_LIBRARY_REAL_NAME  \n
##[PATHSLIST_DEBUG    path1 path2 ...]        \n
##[PATHSLIST_RELEASE  path1 path2 ...]        \n
##[VERBOSE var]                               \n
##          [DEBUG_POSTFIX string]                      \n
##          [RELEASE_POSTFIX string]                    \n
##          [FORCE_DEBUG]                               \n
##          [FORCE_RELEASE]                             \n
## )
##\\endcode
## Brief            : Macro to find a single library.   \\n
## required         : ParseArgumentsMacro.   \\n
## Param 1          : MY_LIBRARY_VARIABLE is the variable you use to contain the lib..   \\n
##                      note1: "${MYLIBRARY}_DEBUG" is set with the "${MYLIBRARYNAME}${CMAKE_DEBUG_POSTFIX}" found.   \\n
##                      note2:  "${MYLIBRARY}" is set with the "${MYLIBRARYNAME}${CMAKE_RELEASE_POSTFIX}" found.   \\n
## Param 2          : MY_LIBRARY_REAL_NAME is the name of the library to find (without postfix).   \\n
## Optional var 1: PATHSLIST_DEBUG paths     -> list of paths in order to find the debug lib.   \\n
## Optional var 2: PATHSLIST_RELEASE paths   -> list of paths in order to find the release lib.   \\n
## Optional var 3: VERBOSE var               -> is the variable which allow the printing messages infos.   \\n
## Optional var 4: DEBUG_POSTFIX string      -> is default to "d".   \\n
## Optional var 5: RELEASE_POSTFIX string    -> is default to empty "".   \\n
## Optional flag 1: FORCE_DEBUG   -> If no DEBUG LIBRARY found, set the debug variable library to the release variable.   \\n
## Optional flag 2: FORCE_RELEASE -> If no RELEASE LIBRARY found, set the release variable library to the debug variable.   \\n
## Usage 1          : FIND_LIB( ${CMAKE_PROJECT_LIB_NAME} ${REAL_PROJECT_LIB_NAME} ).   \\n
## full usage exemple:
##\\code
## FIND_LIB(${CMAKE_SOFA_LIB_NAME} ${REAL_SOFA_LIB_NAME}    \n
##         PATHSLIST_DEBUG                                  \n
##             ${SEARCH_LIB_PATHS}                          \n
##             ${PROJECT_DIR}/lib/Debug                     \n
##             ${PROJECT_DIR}/lib64/Debug                   \n
##         PATHSLIST_RELEASE                                \n
##             ${SEARCH_LIB_PATHS}                          \n
##             ${PROJECT_DIR}/lib/Release                   \n
##             ${PROJECT_DIR}/lib64/Release                 \n
##         VERBOSE          ${PROJECT_VERBOSE}              \n
##         DEBUG_POSTFIX    "d"                             \n
##         RELEASE_POSTFIX  ""                             \n
##         FORCE_DEBUG      true                            \n
##         FORCE_RELEASE    true                            \n
## )
##\\endcode
##
##CMAKE_DOCUMENTATION_END

include(${CMAKE_ROOT}/Modules/CMakeParseArguments.cmake)

MACRO(find_lib MYLIBRARY MYLIBRARYNAME)

    set(options "")
    set(oneValueArgs VERBOSE FORCE_DEBUG DEBUG_POSTFIX FORCE_RELEASE RELEASE_POSTFIX)
    set(multiValueArgs PATHSLIST_DEBUG PATHSLIST_RELEASE)
    cmake_parse_arguments(FIND_LIB "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    # Remain args
    #message("FIND_LIB_UNPARSED_ARGUMENTS = ${FIND_LIB_UNPARSED_ARGUMENTS}")

    # default VERBOSE
    if(NOT DEFINED FIND_LIB_VERBOSE)
        set(FIND_LIB_VERBOSE false)
    endif()

    # default FORCE_DEBUG
    if(NOT DEFINED FIND_LIB_FORCE_DEBUG)
        set(FIND_LIB_FORCE_DEBUG false)
    endif()

    # default FORCE_RELEASE
    if(NOT DEFINED FIND_LIB_FORCE_RELEASE)
        set(FIND_LIB_FORCE_RELEASE false)
    endif()

    # default DEBUG_POSTFIX
    if(NOT DEFINED FIND_LIB_DEBUG_POSTFIX)
        set(FIND_LIB_DEBUG_POSTFIX "d")
    endif()

    # default RELEASE_POSTFIX
    if(NOT DEFINED FIND_LIB_RELEASE_POSTFIX)
        set(FIND_LIB_RELEASE_POSTFIX "")
    endif()

    # search order :
    #message("CMAKE_LIBRARY_PATH = ${CMAKE_LIBRARY_PATH}")
    #message("CMAKE_SYSTEM_LIBRARY_PATH = ${CMAKE_SYSTEM_LIBRARY_PATH}") # can be skip usging: NO_CMAKE_SYSTEM_PATH

    ## Find debug library
    if(FIND_LIB_VERBOSE)
        message(STATUS "\nlooking for debug version: ${MYLIBRARYNAME}${FIND_LIB_DEBUG_POSTFIX}")
    endif()
    find_library("${MYLIBRARY}_DEBUG"
        NAMES
        "${MYLIBRARYNAME}${FIND_LIB_DEBUG_POSTFIX}"
        "${MYLIBRARYNAME}${FIND_LIB_DEBUG_POSTFIX}_${SOFA_VERSION_NUM}"
        PATHS
        ${FIND_LIB_PATHSLIST_DEBUG}
        ${FIND_LIB_UNPARSED_ARGUMENTS}
        )
    if(FIND_LIB_VERBOSE)
        message(STATUS "${MYLIBRARY}_DEBUG = ${${MYLIBRARY}_DEBUG}")
    endif()


    ## Find release library
    if(FIND_LIB_VERBOSE)
        message(STATUS "looking for release version: ${MYLIBRARYNAME}${FIND_LIB_RELEASE_POSTFIX}")
    endif()
    find_library(${MYLIBRARY}
        NAMES
        "${MYLIBRARYNAME}${FIND_LIB_RELEASE_POSTFIX}"
        "${MYLIBRARYNAME}${FIND_LIB_RELEASE_POSTFIX}_${SOFA_VERSION_NUM}"
        PATHS
        ${FIND_LIB_PATHSLIST_RELEASE}
        ${FIND_LIB_UNPARSED_ARGUMENTS}
        )
    if(FIND_LIB_VERBOSE)
        message(STATUS "${MYLIBRARY} = ${${MYLIBRARY}}")
    endif()


    ## Allow to use debug and release version :
    if(FIND_LIB_FORCE_DEBUG)
        ## If no DEBUG LIBRARY found, set the debug variable library to the release variable
        if( NOT ${MYLIBRARY}_DEBUG)
            if(${MYLIBRARY})
                unset(${MYLIBRARY}_DEBUG CACHE)
                set(${MYLIBRARY}_DEBUG ${${MYLIBRARY}} CACHE FILEPATH "Path to a debug library set from release one")
                if(FIND_LIB_VERBOSE)
                    message(STATUS "${MYLIBRARY}_DEBUG NOT FOUND. Set it with the ${MYLIBRARY} content : ${${MYLIBRARY}_DEBUG}")
                endif()
            else()
                if(FIND_LIB_VERBOSE)
                    message("${MYLIBRARY}_DEBUG NOT FOUND.")
                endif()
            endif()
        endif()
    endif()

    if(FIND_LIB_FORCE_RELEASE)
        ## If no RELEASE LIBRARY found, set the release variable library to the debug variable
        if( NOT ${MYLIBRARY})
            if(${MYLIBRARY}_DEBUG)
                unset(${MYLIBRARY} CACHE)
                set(${MYLIBRARY} ${${MYLIBRARY}_DEBUG} CACHE FILEPATH "Path to a release library set from debug one")
                if(FIND_LIB_VERBOSE)
                    message(STATUS "${MYLIBRARY} NOT FOUND. Set it with the ${MYLIBRARY}_DEBUG content : ${{MYLIBRARY}}")
                endif()
            else()
                if(FIND_LIB_VERBOSE)
                    message("${MYLIBRARY} NOT FOUND.")
                endif()
            endif()
        endif()
    endif()

ENDMACRO()
