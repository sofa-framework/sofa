# Adapted version for SOFA of the FindZLIB.cmake file packaged with CMake
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
FindZLIB
--------

Find the native ZLIB includes and library.

Imported Targets
^^^^^^^^^^^^^^^^

.. versionadded:: 3.1

This module defines :prop_tgt:`IMPORTED` target ``ZLIB::ZLIB``, if
ZLIB has been found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``ZLIB_INCLUDE_DIRS``
  where to find zlib.h, etc.
``ZLIB_LIBRARIES``
  List of libraries when using zlib.
``ZLIB_FOUND``
  True if zlib found.
``ZLIB_VERSION``
  .. versionadded:: 3.26
    the version of Zlib found.

  See also legacy variable ``ZLIB_VERSION_STRING``.

.. versionadded:: 3.4
  Debug and Release variants are found separately.

Legacy Variables
^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``ZLIB_VERSION_MAJOR``
  The major version of zlib.

  .. versionchanged:: 3.26
    Superseded by ``ZLIB_VERSION``.
``ZLIB_VERSION_MINOR``
  The minor version of zlib.

  .. versionchanged:: 3.26
    Superseded by ``ZLIB_VERSION``.
``ZLIB_VERSION_PATCH``
  The patch version of zlib.

  .. versionchanged:: 3.26
    Superseded by ``ZLIB_VERSION``.
``ZLIB_VERSION_TWEAK``
  The tweak version of zlib.

  .. versionchanged:: 3.26
    Superseded by ``ZLIB_VERSION``.
``ZLIB_VERSION_STRING``
  The version of zlib found (x.y.z)

  .. versionchanged:: 3.26
    Superseded by ``ZLIB_VERSION``.
``ZLIB_MAJOR_VERSION``
  The major version of zlib.  Superseded by ``ZLIB_VERSION_MAJOR``.
``ZLIB_MINOR_VERSION``
  The minor version of zlib.  Superseded by ``ZLIB_VERSION_MINOR``.
``ZLIB_PATCH_VERSION``
  The patch version of zlib.  Superseded by ``ZLIB_VERSION_PATCH``.

Hints
^^^^^

A user may set ``ZLIB_ROOT`` to a zlib installation root to tell this
module where to look.


#]=======================================================================]

cmake_policy(PUSH)
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.29.0")
  cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>
endif()

if(ZLIB_FIND_COMPONENTS AND NOT ZLIB_FIND_QUIETLY)
  message(AUTHOR_WARNING
    "ZLIB does not provide any COMPONENTS.  Calling\n"
    "  find_package(ZLIB COMPONENTS ...)\n"
    "will always fail."
    )
endif()

set(_ZLIB_SEARCHES)

# Search ZLIB_ROOT first if it is set.
if(ZLIB_ROOT)
  set(_ZLIB_SEARCH_ROOT PATHS ${ZLIB_ROOT} NO_DEFAULT_PATH)
  list(APPEND _ZLIB_SEARCHES _ZLIB_SEARCH_ROOT)
endif()

# Normal search.
set(_ZLIB_x86 "(x86)")
set(_ZLIB_SEARCH_NORMAL
    PATHS "[HKEY_LOCAL_MACHINE\\SOFTWARE\\GnuWin32\\Zlib;InstallPath]"
          "$ENV{ProgramFiles}/zlib"
          "$ENV{ProgramFiles${_ZLIB_x86}}/zlib")
unset(_ZLIB_x86)
list(APPEND _ZLIB_SEARCHES _ZLIB_SEARCH_NORMAL)

set(ZLIB_NAMES z zlib zdll zlib1 zlibstatic zlibwapi zlibvc zlibstat)

# Try each search configuration.
foreach(search ${_ZLIB_SEARCHES})
  find_path(ZLIB_INCLUDE_DIR NAMES zlib.h ${${search}} PATH_SUFFIXES include)
endforeach()

# Allow ZLIB_LIBRARY to be set manually, as the location of the zlib library
if(NOT ZLIB_LIBRARY)
  if(DEFINED CMAKE_FIND_LIBRARY_PREFIXES)
    set(_zlib_ORIG_CMAKE_FIND_LIBRARY_PREFIXES "${CMAKE_FIND_LIBRARY_PREFIXES}")
  else()
    set(_zlib_ORIG_CMAKE_FIND_LIBRARY_PREFIXES)
  endif()
  if(DEFINED CMAKE_FIND_LIBRARY_SUFFIXES)
    set(_zlib_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES "${CMAKE_FIND_LIBRARY_SUFFIXES}")
  else()
    set(_zlib_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES)
  endif()
  # Prefix/suffix of the win32/Makefile.gcc build
  if(WIN32)
    list(APPEND CMAKE_FIND_LIBRARY_PREFIXES "" "lib")
    list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES ".dll.a")
  endif()

  foreach(search ${_ZLIB_SEARCHES})
    find_library(ZLIB_LIBRARY 
      NAMES ${ZLIB_NAMES} 
      NAMES_PER_DIR ${${search}} 
      PATH_SUFFIXES lib
    )
    if(WIN32)
      # possibly also add lib/{win32,win64} to path_suffixes for compat with the windeppack
      find_file(ZLIB_DLL
        NAMES zlib.dll
        PATH_SUFFIXES bin
      )
    endif()
  endforeach()

  # Restore the original find library ordering
  if(DEFINED _zlib_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES)
    set(CMAKE_FIND_LIBRARY_SUFFIXES "${_zlib_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES}")
  else()
    set(CMAKE_FIND_LIBRARY_SUFFIXES)
  endif()
  if(DEFINED _zlib_ORIG_CMAKE_FIND_LIBRARY_PREFIXES)
    set(CMAKE_FIND_LIBRARY_PREFIXES "${_zlib_ORIG_CMAKE_FIND_LIBRARY_PREFIXES}")
  else()
    set(CMAKE_FIND_LIBRARY_PREFIXES)
  endif()
endif()

unset(ZLIB_NAMES)

mark_as_advanced(ZLIB_INCLUDE_DIR ZLIB_LIBRARY)
if(WIN32)
  mark_as_advanced(ZLIB_DLL)
endif()

if(ZLIB_INCLUDE_DIR AND EXISTS "${ZLIB_INCLUDE_DIR}/zlib.h")
  file(STRINGS "${ZLIB_INCLUDE_DIR}/zlib.h" ZLIB_H REGEX "^#define ZLIB_VERSION \"[^\"]*\"$")
  if(ZLIB_H MATCHES "ZLIB_VERSION \"(([0-9]+)\\.([0-9]+)(\\.([0-9]+)(\\.([0-9]+))?)?)")
    set(ZLIB_VERSION_STRING "${CMAKE_MATCH_1}")
    set(ZLIB_VERSION_MAJOR "${CMAKE_MATCH_2}")
    set(ZLIB_VERSION_MINOR "${CMAKE_MATCH_3}")
    set(ZLIB_VERSION_PATCH "${CMAKE_MATCH_5}")
    set(ZLIB_VERSION_TWEAK "${CMAKE_MATCH_7}")
  else()
    set(ZLIB_VERSION_STRING "")
    set(ZLIB_VERSION_MAJOR "")
    set(ZLIB_VERSION_MINOR "")
    set(ZLIB_VERSION_PATCH "")
    set(ZLIB_VERSION_TWEAK "")
  endif()
  set(ZLIB_MAJOR_VERSION "${ZLIB_VERSION_MAJOR}")
  set(ZLIB_MINOR_VERSION "${ZLIB_VERSION_MINOR}")
  set(ZLIB_PATCH_VERSION "${ZLIB_VERSION_PATCH}")
  set(ZLIB_VERSION "${ZLIB_VERSION_STRING}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ZLIB REQUIRED_VARS ZLIB_LIBRARY ZLIB_INCLUDE_DIR
                                       VERSION_VAR ZLIB_VERSION
                                       HANDLE_COMPONENTS)

if(ZLIB_FOUND)
    set(ZLIB_INCLUDE_DIRS ${ZLIB_INCLUDE_DIR})

    if(NOT ZLIB_LIBRARIES)
      set(ZLIB_LIBRARIES ${ZLIB_LIBRARY})
    endif()

    if(NOT TARGET ZLIB::ZLIB)
      # Do not use UNKNOWN library type here as link libraries would 
      # use the IMPORTED_LOCATION library for linking, so we force
      # shared ZLIB
      add_library(ZLIB::ZLIB SHARED IMPORTED)
      set_target_properties(ZLIB::ZLIB PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${ZLIB_INCLUDE_DIRS}")
      if(WIN32)
        set_target_properties(ZLIB::ZLIB PROPERTIES
          IMPORTED_LOCATION "${ZLIB_DLL}"
          IMPORTED_IMPLIB "${ZLIB_LIBRARY}")
      else()
        set_target_properties(ZLIB::ZLIB PROPERTIES
          IMPORTED_LOCATION "${ZLIB_LIBRARY}")
      endif()

    endif()
endif()

cmake_policy(POP)
