# Find the metis headers and libraries
# Behavior is to first look for config files, such as the one installed by some package
# managers who provides their own cmake files.
# Most of them and official sources does not provide cmake finders, so if no config files
# were found, this tries to find the library by looking at headers / lib file.
#
# Defines:
#   metis_FOUND : True if metis is found
#   metis_VERSION : metis version if found
#
# Provides both targets metis and metis::metis.
#   Target metis::metis is just an alias to metis.
# We chose to create an alias to provide a unified interface usable whatever the package manager
# was used to provide the library, as some package managers (such vcpkg) defines only short name
# for the target, whereas others (such as conan) defines a fully qualified name.

find_package(metis NO_MODULE QUIET HINTS ${metis_DIR} NAMES metis Metis)


if(NOT metis_FIND_VERSION)
  if(NOT metis_FIND_VERSION_MAJOR)
    set(metis_FIND_VERSION_MAJOR 0)
  endif(NOT metis_FIND_VERSION_MAJOR)
  if(NOT metis_FIND_VERSION_MINOR)
    set(metis_FIND_VERSION_MINOR 0)
  endif(NOT metis_FIND_VERSION_MINOR)
  if(NOT metis_FIND_VERSION_PATCH)
    set(metis_FIND_VERSION_PATCH 0)
  endif(NOT metis_FIND_VERSION_PATCH)
  set(metis_FIND_VERSION "${metis_FIND_VERSION_MAJOR}.${metis_FIND_VERSION_MINOR}.${metis_FIND_VERSION_PATCH}")
endif()

macro(_metis_check_version)
  if(EXISTS "${metis_INCLUDE_DIR}/metis.h")
    file(READ "${metis_INCLUDE_DIR}/metis.h" _metis_version_header)
  endif()

  string(REGEX MATCH "define[ \t]+METIS_VER_MAJOR[ \t]+([0-9]+)" _metis_major_version_match "${_metis_version_header}")
  set(metis_VERSION_MAJOR "${CMAKE_MATCH_1}")
  string(REGEX MATCH "define[ \t]+METIS_VER_MINOR[ \t]+([0-9]+)" _metis_minor_version_match "${_metis_version_header}")
  set(metis_VERSION_MINOR "${CMAKE_MATCH_1}")
  string(REGEX MATCH "define[ \t]+METIS_VER_SUBMINOR[ \t]+([0-9]+)" _metis_patch_version_match "${_metis_version_header}")
  set(metis_VERSION_PATCH "${CMAKE_MATCH_1}")

  set(metis_VERSION ${metis_VERSION_MAJOR}.${metis_VERSION_MINOR}.${metis_VERSION_PATCH})
  set(metis_VERSION_OK TRUE)
  if(${metis_VERSION} VERSION_LESS ${metis_FIND_VERSION})
    set(metis_VERSION_OK FALSE)
    message(SEND_ERROR "metis version ${metis_VERSION} found in ${metis_INCLUDE_DIR}, "
                       "but at least version ${metis_FIND_VERSION} is required")
  endif()
  if(${metis_FIND_VERSION_EXACT} AND NOT ${metis_VERSION} VERSION_EQUAL ${metis_FIND_VERSION})
    set(metis_VERSION_OK FALSE)
    message(SEND_ERROR "metis version ${metis_VERSION} found in ${metis_INCLUDE_DIR}, "
                       "but exact version ${metis_FIND_VERSION} is required")
  endif()
endmacro()

if(TARGET metis)
  set(metis_FOUND TRUE) # only metis_FOUND has been set
  if(metis_INCLUDE_DIR AND NOT DEFINED metis_VERSION)
    _metis_check_version()
    set(metis_FOUND ${metis_VERSION_OK})
  endif()
  add_library(metis::metis ALIAS metis)
else()

  if(NOT metis_INCLUDE_DIR)
    find_path(metis_INCLUDE_DIR
      NAMES metis.h
      PATH_SUFFIXES include
    )
  endif()

  if(NOT metis_LIBRARY)
    find_library(metis_LIBRARY
      NAMES metis
      PATH_SUFFIXES lib
    )
  endif()

  if(metis_INCLUDE_DIR AND metis_LIBRARY)
    _metis_check_version()
    set(metis_FOUND ${metis_VERSION_OK})
  endif()

  if(metis_FOUND)
    set(metis_LIBRARIES ${metis_LIBRARY})
    set(metis_INCLUDE_DIRS ${metis_INCLUDE_DIR})

    if(NOT metis_FIND_QUIETLY)
      message(STATUS "Found metis: ${metis_LIBRARIES} (version ${metis_VERSION} from ${metis_INCLUDE_DIR}/metis.h)")
    endif()

    if(NOT TARGET metis)
      add_library(metis INTERFACE IMPORTED)
      set_property(TARGET metis PROPERTY INTERFACE_LINK_LIBRARIES "${metis_LIBRARIES}")
      set_property(TARGET metis PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${metis_INCLUDE_DIR}")
    endif()
    add_library(metis::metis ALIAS metis)
  else()
  endif()
  mark_as_advanced(metis_INCLUDE_DIR metis_LIBRARY)
endif()

if(NOT metis_FOUND AND metis_FIND_REQUIRED)
  message(FATAL_ERROR "Cannot find metis")
endif()
