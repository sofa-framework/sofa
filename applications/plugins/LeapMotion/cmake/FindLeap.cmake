# Find Leap
#
# Finds the libraries and header files for the Leap SDK for Leap Motion's
# hand tracker.
#
# This module defines
# LEAP_FOUND       - Leap was found
# LEAP_INCLUDE_DIR - Directory containing LEAP header files
# LEAP_LIBRARY     - Library name of Leap library
# LEAP_INCLUDE_DIRS- Directories required by CMake convention
# LEAP_LIBRARIES   - Libraries required by CMake convention
#
# Based on the FindFMODEX.cmake of Team Pantheon.

# Don't be verbose if previously run successfully
IF(LEAP_INCLUDE_DIR AND LEAP_LIBRARY)
    SET(LEAP_FIND_QUIETLY TRUE)
ENDIF(LEAP_INCLUDE_DIR AND LEAP_LIBRARY)

# Set locations to search
IF(UNIX)
    SET(LEAP_INCLUDE_SEARCH_DIRS
        /usr/include
        /usr/local/include
        /opt/leap/include
        /opt/leap_sdk/include
        /opt/include INTERNAL)
    SET(LEAP_LIBRARY_SEARCH_DIRS
        /usr/lib
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64
        /opt/leap/lib
        /opt/leap/lib64
        /opt/leap_sdk/lib
        /opt/leap_sdk/lib64 INTERNAL)
    SET(LEAP_INC_DIR_SUFFIXES PATH_SUFFIXES leap)
ELSE(UNIX)
    #WIN32
    SET(LEAP_INC_DIR_SUFFIXES PATH_SUFFIXES inc)
    SET(LEAP_LIB_DIR_SUFFIXES PATH_SUFFIXES lib)
ENDIF(UNIX)

# Set name of the Leap library to use
IF(APPLE)
    SET(LEAP_LIBRARY_NAME libLeap.dylib)
ELSE(APPLE)
    IF(UNIX)
        SET(LEAP_LIBRARY_NAME libLeap.so)
    ELSE(UNIX)
        # TODO Different libraries are provided for compile and runtime
        SET(LEAP_LIBRARY_NAME libLeap.lib)
    ENDIF(UNIX)
ENDIF(APPLE)

IF(NOT LEAP_FIND_QUIETLY)
    MESSAGE(STATUS "Checking for Leap")
ENDIF(NOT LEAP_FIND_QUIETLY)

# Search for header files
FIND_PATH(LEAP_INCLUDE_DIR Leap.h
    PATHS ${LEAP_INCLUDE_SEARCH_PATHS}
    PATH_SUFFIXES ${LEAP_INC_DIR_SUFFIXES})

# Search for library
FIND_LIBRARY(LEAP_LIBRARY ${LEAP_LIBRARY_NAME}
    PATHS ${LEAP_LIBRARY_SEARCH_DIRS}
    PATH_SUFFIXES ${LEAP_LIB_DIR_SUFFIXES})

SET(LEAP_INCLUDE_DIR ${LEAP_INCLUDE_DIR} CACHE STRING
    "Directory containing LEAP header files")
SET(LEAP_LIBRARY ${LEAP_LIBRARY} CACHE STRING "Library name of Leap library")

SET(LEAP_INCLUDE_DIRS ${LEAP_INCLUDE_DIR} )
SET(LEAP_LIBRARIES ${LEAP_LIBRARY} )

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(Leap DEFAULT_MSG LEAP_LIBRARY LEAP_INCLUDE_DIR)

MARK_AS_ADVANCED(LEAP_INCLUDE_DIR LEAP_LIBRARY)

