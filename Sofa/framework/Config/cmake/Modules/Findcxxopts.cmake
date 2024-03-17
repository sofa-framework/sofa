# Find the cxxopts headers
# Behavior is to first look for config files, such as the one installed by some package
# managers who provides their own cmake files.
# Most of them and official sources does not provide cmake finders, so if no config files
# were found, this tries to find the library by looking at headers file.
#
# Defines:
#   cxxopts_FOUND : True if cxxopts is found
#   cxxopts_VERSION : cxxopts version if found
#
# Provides both target cxxopts::cxxopts.

find_package(cxxopts NO_MODULE QUIET HINTS ${cxxopts_DIR})

macro(_cxxopts_check_version)
  if(EXISTS "${cxxopts_INCLUDE_DIR}/cxxopts.hpp")
   file(READ "${cxxopts_INCLUDE_DIR}/cxxopts.hpp" _cxxopts_version_header)
  endif()

  string(REGEX MATCH "define[ \t]+CXXOPTS__VERSION_MAJOR[ \t]+([0-9]+)" _cxxopts_major_version_match "${_cxxopts_version_header}")
  set(cxxopts_VERSION_MAJOR "${CMAKE_MATCH_1}")
  string(REGEX MATCH "define[ \t]+CXXOPTS__VERSION_MINOR[ \t]+([0-9]+)" _cxxopts_minor_version_match "${_cxxopts_version_header}")
  set(cxxopts_VERSION_MINOR "${CMAKE_MATCH_1}")
  string(REGEX MATCH "define[ \t]+CXXOPTS__VERSION_PATCH[ \t]+([0-9]+)" _cxxopts_patch_version_match "${_cxxopts_version_header}")
  set(cxxopts_VERSION_PATCH "${CMAKE_MATCH_1}")

  set(cxxopts_VERSION ${cxxopts_VERSION_MAJOR}.${cxxopts_VERSION_MINOR}.${cxxopts_VERSION_PATCH})
  set(cxxopts_VERSION_OK TRUE)
  if(${cxxopts_VERSION} VERSION_LESS ${cxxopts_FIND_VERSION})
    set(cxxopts_VERSION_OK FALSE)
    message(WARNING "cxxopts version ${cxxopts_VERSION} found in ${cxxopts_INCLUDE_DIR}/cxxopts.hpp, "
                        "but at least version ${cxxopts_FIND_VERSION} is required")
  endif()
  if(${cxxopts_FIND_VERSION_EXACT} AND NOT ${cxxopts_VERSION} VERSION_EQUAL ${cxxopts_FIND_VERSION})
    set(cxxopts_VERSION_OK FALSE)
    message(WARNING "cxxopts version ${cxxopts_VERSION} found in ${cxxopts_INCLUDE_DIR}, "
                        "but exact version ${cxxopts_FIND_VERSION_EXACT} is required")
  endif()
endmacro()

if(NOT TARGET cxxopts::cxxopts)

  if(NOT cxxopts_INCLUDE_DIR)
    find_path(cxxopts_INCLUDE_DIR
      NAMES cxxopts.hpp
      PATH_SUFFIXES include
    )
  endif()

  if(cxxopts_INCLUDE_DIR)
    if(cxxopts_FIND_VERSION)
      _cxxopts_check_version()
      set(cxxopts_FOUND ${cxxopts_VERSION_OK})
    else()
      set(cxxopts_FOUND TRUE)
    endif()
  endif()

  if(cxxopts_FOUND)
    set(cxxopts_INCLUDE_DIRS ${cxxopts_INCLUDE_DIR})

    if(NOT cxxopts_FIND_QUIETLY)
      message(STATUS "Found cxxopts version ${cxxopts_VERSION} from ${cxxopts_INCLUDE_DIR}/cxxopts.hpp)")
    endif()

    if(NOT TARGET cxxopts::cxxopts)
      add_library(cxxopts::cxxopts INTERFACE IMPORTED)
      set_property(TARGET cxxopts::cxxopts PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${cxxopts_INCLUDE_DIR}")
    endif()
  else()
  endif()
  
  mark_as_advanced(cxxopts_INCLUDE_DIR)
endif()

if(NOT cxxopts_FOUND AND cxxopts_FIND_REQUIRED)
  message(FATAL_ERROR "Cannot find cxxopts")
endif()
