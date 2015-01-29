# - Find the libfreenect includes and library
# This module defines
# FREENECT_INCLUDE_DIR, path to libfreenect.h, etc.
# FREENECT_LIBRARIES, the libraries required to use FREENECT.
# FREENECT_FOUND, If false, do not try to use FREENECT.
# also defined, but not for general use are
# FREENECT_freenect_LIBRARY, where to find the FREENECT library.

find_path(FREENECT_INCLUDE_DIR libfreenect.h
    /usr/include
    /usr/include/libfreenect
    /usr/local/include
    /usr/local/include/libfreenect)

find_library(FREENECT_LIBRARY freenect
    /usr/lib
    /usr/local/lib)


mark_as_advanced(FREENECT_INCLUDE_DIR FREENECT_freenect_LIBRARY)

set(FREENECT_FOUND "NO")
if(FREENECT_INCLUDE_DIR)
    if(FREENECT_LIBRARY)
        set(FREENECT_FOUND "YES")
        set(FREENECT_LIBRARIES "${FREENECT_LIBRARY}")
    endif()
endif()

if(FREENECT_FOUND)
    message(STATUS "Found freenect library")
else(FREENECT_FOUND)
    if(FREENECT_FIND_REQUIRED)
        message(FATAL_ERROR "Could not find libfreenect")
    else(FREENECT_FIND_REQUIRED)
        message(STATUS "Could not find libfreenect")
    endif()
endif()
