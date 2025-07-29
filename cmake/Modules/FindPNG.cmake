# Adapted version for SOFA of the FindPNG.cmake file packaged with CMake
# Modifications: 
#  - look for DLL binary files on Windows and 
# set the target properties correctly to allow installation to copy them for 
# our packaging needs. 
#  - disable look for static lib (as add_library must be set to SHARED instead of UNKNOWN)
#  - discard specific build configurations (Debug, Release) 
#
# --------
#
# Distributed under the OSI-approved BSD 3-Clause License. 
# Copyright 2000-2025 Kitware, Inc. and Contributors
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
# Neither the name of Kitware, Inc. nor the names of Contributors
# may be used to endorse or promote products derived from this
# software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#[=======================================================================[.rst:
FindPNG
-------

Find libpng, the official reference library for the PNG image format.

Imported Targets
^^^^^^^^^^^^^^^^

.. versionadded:: 3.5

This module defines the following :prop_tgt:`IMPORTED` target:

``PNG::PNG``
  The libpng library, if found.

Result variables
^^^^^^^^^^^^^^^^

This module will set the following variables in your project:

``PNG_INCLUDE_DIRS``
  where to find png.h, etc.
``PNG_LIBRARIES``
  the libraries to link against to use PNG.
``PNG_DEFINITIONS``
  You should add_definitions(${PNG_DEFINITIONS}) before compiling code
  that includes png library files.
``PNG_FOUND``
  If false, do not try to use PNG.
``PNG_VERSION_STRING``
  the version of the PNG library found (since CMake 2.8.8)

Obsolete variables
^^^^^^^^^^^^^^^^^^

The following variables may also be set, for backwards compatibility:

``PNG_LIBRARY``
  where to find the PNG library.
``PNG_INCLUDE_DIR``
  where to find the PNG headers (same as PNG_INCLUDE_DIRS)

Since PNG depends on the ZLib compression library, none of the above
will be defined unless ZLib can be found.
#]=======================================================================]

cmake_policy(PUSH)
cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>

if(NOT TARGET ZLIB::ZLIB)
  if(PNG_FIND_QUIETLY)
    set(_FIND_ZLIB_ARG QUIET)
  endif()
  find_package(ZLIB ${_FIND_ZLIB_ARG})
endif()

if(ZLIB_FOUND)
  set(_PNG_VERSION_SUFFIXES 17 16 15 14 12)

  list(APPEND _PNG_INCLUDE_PATH_SUFFIXES include/libpng)
  foreach(v IN LISTS _PNG_VERSION_SUFFIXES)
    list(APPEND _PNG_INCLUDE_PATH_SUFFIXES include/libpng${v})
  endforeach()

  find_path(PNG_PNG_INCLUDE_DIR png.h PATH_SUFFIXES ${_PNG_INCLUDE_PATH_SUFFIXES} PATHS ${_PNG_INCLUDE_SEARCH_NORMAL} )
  mark_as_advanced(PNG_PNG_INCLUDE_DIR)

  list(APPEND PNG_NAMES png libpng)
  if (PNG_FIND_VERSION MATCHES "^([0-9]+)\\.([0-9]+)(\\..*)?$")
    set(_PNG_VERSION_SUFFIX_MIN "${CMAKE_MATCH_1}${CMAKE_MATCH_2}")
    if (PNG_FIND_VERSION_EXACT)
      set(_PNG_VERSION_SUFFIXES ${_PNG_VERSION_SUFFIX_MIN})
    else ()
      string(REGEX REPLACE
          "${_PNG_VERSION_SUFFIX_MIN}.*" "${_PNG_VERSION_SUFFIX_MIN}"
          _PNG_VERSION_SUFFIXES "${_PNG_VERSION_SUFFIXES}")
    endif ()
    unset(_PNG_VERSION_SUFFIX_MIN)
  endif ()
  foreach(v IN LISTS _PNG_VERSION_SUFFIXES)
    list(APPEND PNG_NAMES png${v} libpng${v} libpng${v}_static)
  endforeach()
  unset(_PNG_VERSION_SUFFIXES)
  # For compatibility with versions prior to this multi-config search, honor
  # any PNG_LIBRARY that is already specified and skip the search.
  if(NOT PNG_LIBRARY)
    find_library(PNG_LIBRARY NAMES ${PNG_NAMES} NAMES_PER_DIR PATH_SUFFIXES lib)
    mark_as_advanced(PNG_LIBRARY)
  endif()
  # look for DLL file on Windows for installation
  if(WIN32)
    foreach(v IN LISTS PNG_NAMES)
      list(APPEND PNG_DLL_FILENAMES ${v}.dll)
    endforeach()
    find_file(PNG_DLL NAMES ${PNG_DLL_FILENAMES} PATH_SUFFIXES bin)
    mark_as_advanced(PNG_DLL)
 endif()
  unset(PNG_NAMES)
  unset(PNG_DLL_FILENAMES)
  unset(_PNG_INCLUDE_PATH_SUFFIXES)

  # Set by select_library_configurations(), but we want the one from
  # find_package_handle_standard_args() below.
  unset(PNG_FOUND)

  if (PNG_LIBRARY AND PNG_PNG_INCLUDE_DIR)
      # png.h includes zlib.h. Sigh.
      set(PNG_INCLUDE_DIRS ${PNG_PNG_INCLUDE_DIR} ${ZLIB_INCLUDE_DIR} )
      set(PNG_INCLUDE_DIR ${PNG_INCLUDE_DIRS} ) # for backward compatibility
      set(PNG_LIBRARIES ${PNG_LIBRARY} ${ZLIB_LIBRARY})
      if((CMAKE_SYSTEM_NAME STREQUAL "Linux") AND
         ("${PNG_LIBRARY}" MATCHES "\\${CMAKE_STATIC_LIBRARY_SUFFIX}$"))
        list(APPEND PNG_LIBRARIES m)
      endif()

      if(NOT TARGET PNG::PNG)
        add_library(PNG::PNG SHARED IMPORTED)
        set_target_properties(PNG::PNG PROPERTIES
          INTERFACE_COMPILE_DEFINITIONS "${_PNG_COMPILE_DEFINITIONS}"
          INTERFACE_INCLUDE_DIRECTORIES "${PNG_INCLUDE_DIRS}"
          INTERFACE_LINK_LIBRARIES ZLIB::ZLIB)
        if((CMAKE_SYSTEM_NAME STREQUAL "Linux") AND
           ("${PNG_LIBRARY}" MATCHES "\\${CMAKE_STATIC_LIBRARY_SUFFIX}$"))
          set_property(TARGET PNG::PNG APPEND PROPERTY
            INTERFACE_LINK_LIBRARIES m)
        endif()

        if(WIN32)
          set_target_properties(PNG::PNG PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGES "C"
            IMPORTED_LOCATION "${PNG_DLL}"
            IMPORTED_IMPLIB "${PNG_LIBRARY}")
        else()
          set_target_properties(PNG::PNG PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGES "C"
            IMPORTED_LOCATION "${PNG_LIBRARY}")
        endif()
        
      endif()

      unset(_PNG_COMPILE_DEFINITIONS)
  endif ()

  if (PNG_PNG_INCLUDE_DIR AND EXISTS "${PNG_PNG_INCLUDE_DIR}/png.h")
      file(STRINGS "${PNG_PNG_INCLUDE_DIR}/png.h" png_version_str REGEX "^#define[ \t]+PNG_LIBPNG_VER_STRING[ \t]+\".+\"")

      string(REGEX REPLACE "^#define[ \t]+PNG_LIBPNG_VER_STRING[ \t]+\"([^\"]+)\".*" "\\1" PNG_VERSION_STRING "${png_version_str}")
      unset(png_version_str)
  endif ()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PNG
                                  REQUIRED_VARS PNG_LIBRARY PNG_PNG_INCLUDE_DIR
                                  VERSION_VAR PNG_VERSION_STRING)

cmake_policy(POP)
