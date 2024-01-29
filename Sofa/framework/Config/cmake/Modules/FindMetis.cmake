# Find the metis headers and libraries
# Behavior is to first look for config files, such as the one installed by some package
# managers who provides their own cmake files.
# Most of them and official sources does not provide cmake finders, so if no config files
# were found, this tries to find the library by looking at headers / lib file.
#
# Defines:
#   Metis_FOUND : True if metis is found
#   Metis_FOUND : True if metis is found
#
# Provides both targets metis and metis::metis.
#   Target metis::metis is just an alias to metis.
# We chose to create an alias to provide a unified interface usable whatever the package manager
# was used to provide the library, as some package managers (such vcpkg) defines only short name
# for the target, whereas others (such as conan) defines a fully qualified name.

find_package(metis NO_MODULE QUIET)

if(NOT Metis_FIND_VERSION)
  if(NOT Metis_FIND_VERSION_MAJOR)
    set(Metis_FIND_VERSION_MAJOR 0)
  endif(NOT Metis_FIND_VERSION_MAJOR)
  if(NOT Metis_FIND_VERSION_MINOR)
    set(Metis_FIND_VERSION_MINOR 0)
  endif(NOT Metis_FIND_VERSION_MINOR)
  if(NOT Metis_FIND_VERSION_PATCH)
    set(Metis_FIND_VERSION_PATCH 0)
  endif(NOT Metis_FIND_VERSION_PATCH)
  set(Metis_FIND_VERSION "${Metis_FIND_VERSION_MAJOR}.${Metis_FIND_VERSION_MINOR}.${Metis_FIND_VERSION_PATCH}")
endif()

macro(_metis_check_version)
  if(EXISTS "${Metis_INCLUDE_DIR}/metis.h")
    file(READ "${Metis_INCLUDE_DIR}/metis.h" _metis_version_header)
  endif()

  string(REGEX MATCH "define[ \t]+METIS_VER_MAJOR[ \t]+([0-9]+)" _metis_major_version_match "${_metis_version_header}")
  set(Metis_VERSION_MAJOR "${CMAKE_MATCH_1}")
  string(REGEX MATCH "define[ \t]+METIS_VER_MINOR[ \t]+([0-9]+)" _metis_minor_version_match "${_metis_version_header}")
  set(Metis_VERSION_MINOR "${CMAKE_MATCH_1}")
  string(REGEX MATCH "define[ \t]+METIS_VER_SUBMINOR[ \t]+([0-9]+)" _metis_patch_version_match "${_metis_version_header}")
  set(Metis_VERSION_PATCH "${CMAKE_MATCH_1}")

  set(Metis_VERSION ${Metis_VERSION_MAJOR}.${Metis_VERSION_MINOR}.${Metis_VERSION_PATCH})
  set(Metis_VERSION_OK TRUE)
  if(${Metis_VERSION} VERSION_LESS ${Metis_FIND_VERSION})
    set(Metis_VERSION_OK FALSE)
    message(SEND_ERROR "Metis version ${Metis_VERSION} found in ${Metis_INCLUDE_DIR}, "
                       "but at least version ${Metis_FIND_VERSION} is required")
  endif()
  if(${Metis_FIND_VERSION_EXACT} AND NOT ${Metis_VERSION} VERSION_EQUAL ${Metis_FIND_VERSION})
    set(Metis_VERSION_OK FALSE)
    message(SEND_ERROR "Metis version ${Metis_VERSION} found in ${Metis_INCLUDE_DIR}, "
                       "but exact version ${Metis_FIND_VERSION} is required")
  endif()
  # message(STATUS "Metis version found: ${Metis_VERSION} in ${Metis_INCLUDE_DIR}, ${Metis_FIND_VERSION} was required ")
endmacro()

if(TARGET metis)
  set(Metis_FOUND TRUE) # only metis_FOUND has been set
  if(Metis_INCLUDE_DIR AND NOT DEFINED Metis_VERSION)
    _metis_check_version()
  endif()
  set(Metis_FOUND ${Metis_VERSION_OK})
  add_library(metis::metis ALIAS metis)
else()

  if(NOT Metis_INCLUDE_DIR)
    find_path(Metis_INCLUDE_DIR
      NAMES metis.h
      PATH_SUFFIXES include
    )
  endif()

  if(NOT Metis_LIBRARY)
    find_library(Metis_LIBRARY
      NAMES metis
      PATH_SUFFIXES lib
    )
  endif()

  if(Metis_INCLUDE_DIR AND Metis_LIBRARY)
    _metis_check_version()
    set(Metis_FOUND ${Metis_VERSION_OK})
  endif()

  if(Metis_FOUND)
    set(Metis_LIBRARIES ${Metis_LIBRARY})
    set(Metis_INCLUDE_DIRS ${Metis_INCLUDE_DIR})

    if(NOT Metis_FIND_QUIETLY)
      message(STATUS "Found Metis: ${Metis_LIBRARIES} (version ${Metis_VERSION} from ${Metis_INCLUDE_DIR}/metis.h)")
    endif()

    if(NOT TARGET metis)
      add_library(metis INTERFACE IMPORTED)
      set_property(TARGET metis PROPERTY INTERFACE_LINK_LIBRARIES "${Metis_LIBRARIES}")
      set_property(TARGET metis PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${Metis_INCLUDE_DIR}")
    endif()
    add_library(metis::metis ALIAS metis)
  else()
  endif()
  mark_as_advanced(Metis_INCLUDE_DIR Metis_LIBRARY)
endif()

if(NOT Metis_FOUND AND Metis_FIND_REQUIRED)
  message(FATAL_ERROR "Cannot find metis")
endif()