# - Try to find Nlohmann Json lib
#
# This module supports requiring a minimum version, e.g. you can do
#   find_package(Json 3.1.2)
# to require version 3.1.2 or newer of Json.
#
# This module reads hints about search locations from variables:
#  JSON_ROOT - Preferred installation prefix
#
# Once done this will define
#
#  JSON_FOUND - system has json lib with correct version
#  JSON_INCLUDE_DIR - the json include directory
#  JSON_VERSION - json version
include_guard(GLOBAL)

if(NOT Json_FIND_VERSION)
    if(NOT Json_FIND_VERSION_MAJOR)
        set(Json_FIND_VERSION_MAJOR 0)
    endif(NOT Json_FIND_VERSION_MAJOR)
    if(NOT Json_FIND_VERSION_MINOR)
        set(Json_FIND_VERSION_MINOR 0)
    endif(NOT Json_FIND_VERSION_MINOR)
    if(NOT Json_FIND_VERSION_PATCH)
        set(Json_FIND_VERSION_PATCH 0)
    endif(NOT Json_FIND_VERSION_PATCH)
    set(Json_FIND_VERSION "${Json_FIND_VERSION_MAJOR}.${Json_FIND_VERSION_MINOR}.${Json_FIND_VERSION_PATCH}")
endif()

macro(_json_check_version)
    foreach(version_header nlohmann_json.h json.h json.hpp)
        if(EXISTS "${JSON_INCLUDE_DIR}/${version_header}")
            file(READ "${JSON_INCLUDE_DIR}/${version_header}" _json_version_header)
            break()
        endif()
    endforeach()

    string(REGEX MATCH "define[ \t]+NLOHMANN_JSON_VERSION_MAJOR[ \t]+([0-9]+)" _json_major_version_match "${_json_version_header}")
    set(JSON_MAJOR_VERSION "${CMAKE_MATCH_1}")
    string(REGEX MATCH "define[ \t]+NLOHMANN_JSON_VERSION_MINOR[ \t]+([0-9]+)" _json_minor_version_match "${_json_version_header}")
    set(JSON_MINOR_VERSION "${CMAKE_MATCH_1}")
    string(REGEX MATCH "define[ \t]+NLOHMANN_JSON_VERSION_PATCH[ \t]+([0-9]+)" _json_patch_version_match "${_json_version_header}")
    set(JSON_PATCH_VERSION "${CMAKE_MATCH_1}")

    set(JSON_VERSION ${JSON_MAJOR_VERSION}.${JSON_MINOR_VERSION}.${JSON_PATCH_VERSION})
    if(${JSON_VERSION} VERSION_LESS ${Json_FIND_VERSION})
        set(JSON_VERSION_OK FALSE)
    else()
        set(JSON_VERSION_OK TRUE)
    endif()

    if(NOT JSON_VERSION_OK)
        message(STATUS "Json version ${JSON_VERSION} found in ${JSON_INCLUDE_DIR}, "
                       "but at least version ${Json_FIND_VERSION} is required")
    endif()
endmacro()

if(JSON_INCLUDE_DIR)
    # Directory already in cache and existing
    _json_check_version()
    set(JSON_FOUND ${JSON_VERSION_OK})
else()
    find_path(JSON_INCLUDE_DIR NAMES nlohmann_json.h json.h json.hpp
        PATHS
            ${JSON_ROOT}
            ${CMAKE_INSTALL_PREFIX}/include
        PATH_SUFFIXES
            nlohmann
            include/nlohmann
        # If cross-compiling and typically use CMAKE_FIND_ROOT_PATH variable,
        # each of its directory entry will be prepended to PATHS locations, and
        # JSON_ROOT is set as an absolute path. So we have to disable this behavior
        # for such external libs
        NO_CMAKE_FIND_ROOT_PATH
        )
        
    if(JSON_INCLUDE_DIR)
        _json_check_version()
    endif()

    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(Json DEFAULT_MSG JSON_INCLUDE_DIR JSON_VERSION_OK)

    mark_as_advanced(JSON_INCLUDE_DIR)
endif()

