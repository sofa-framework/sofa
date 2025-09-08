#[=======================================================================[.rst:
FindCImg
-------

Finds the CImg library.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following imported targets, if found:

``CImg``
  The CImg library

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``CImg_FOUND``
  True if the system has the CImg library.
``CImg_VERSION``
  The version of the CImg library which was found.
``CImg_INCLUDE_DIRS``
  Include directories needed to use CImg.


Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``CImg_INCLUDE_DIR``
  The directory containing ``CImg.h``.

#]=======================================================================]

cmake_minimum_required(VERSION 3.9)

if(NOT "${CIMG_H_DIR}" STREQUAL "" AND NOT EXISTS "${CIMG_H_DIR}/CImg.h")
    unset(CIMG_H_DIR CACHE)
endif()
if(NOT "${CIMG_PLUGINS_H_DIR}" STREQUAL "" AND NOT EXISTS "${CIMG_H_DIR}/plugins/skeleton.h")
    unset(CIMG_PLUGINS_H_DIR CACHE)
endif()

find_path(CIMG_H_DIR
    NAMES CImg.h
    HINTS ${CMAKE_INSTALL_PREFIX} ${CIMG_DIR}
    PATH_SUFFIXES include include/linux
)

find_path(CIMG_PLUGINS_H_DIR
    NAMES plugins/skeleton.h
    HINTS ${CMAKE_INSTALL_PREFIX} ${CIMG_DIR}
    PATH_SUFFIXES include/CImg include/linux/CImg
)

if(NOT CIMG_H_DIR OR NOT CIMG_PLUGINS_H_DIR)
    set(CImg_FOUND FALSE)
else()
    set(CImg_FOUND TRUE)

    set(CImg_INCLUDE_DIR ${CIMG_H_DIR} CACHE STRING "")
    set(CImg_PLUGINS_INCLUDE_DIR ${CIMG_PLUGINS_H_DIR} CACHE STRING "")
    set(CImg_INCLUDE_DIRS ${CImg_INCLUDE_DIR} ${CIMG_PLUGINS_H_DIR})

    file(READ "${CIMG_H_DIR}/CImg.h" header)
    string(REGEX MATCH "#define cimg_version ([0-9a-zA-Z\.]+)" _ "${header}")
    set(CImg_VERSION "${CMAKE_MATCH_1}")

    if(NOT TARGET CImg)
        add_library(CImg INTERFACE)
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CImg
  FOUND_VAR CImg_FOUND
  REQUIRED_VARS
    CImg_INCLUDE_DIRS
  VERSION_VAR CImg_VERSION
)
