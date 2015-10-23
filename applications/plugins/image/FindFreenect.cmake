# - Find the libfreenect includes and library
# This module defines
# FREENECT_INCLUDE_DIR, path to libfreenect.h, etc.
# FREENECT_LIBRARIES, the libraries required to use FREENECT.
# FREENECT_FOUND, If false, do not try to use FREENECT.

find_path(FREENECT_INCLUDE_DIR libfreenect.h
    /usr/include
    /usr/include/libfreenect
    /usr/local/include
    /usr/local/include/libfreenect)

find_library(FREENECT_LIBRARY freenect
    /usr/lib
    /usr/local/lib)

mark_as_advanced(FREENECT_INCLUDE_DIR)
mark_as_advanced(FREENECT_LIBRARY)

set(FREENECT_FOUND "NO")
if(FREENECT_INCLUDE_DIR)
    if(FREENECT_LIBRARY)
        set(FREENECT_FOUND "YES")
        set(FREENECT_LIBRARIES "${FREENECT_LIBRARY}")
    endif()
endif()

if(FREENECT_FOUND)
    if(NOT Freenect_FIND_QUIETLY)
        message(STATUS "Found freenect library")
    endif()
else(FREENECT_FOUND)
    if(Freenect_FIND_REQUIRED)
        message(FATAL_ERROR "Could not find the freenect library")
    else()
        if(NOT Freenect_FIND_QUIETLY)
             message(STATUS "Could not find the freenect library")
        endif()
    endif()
endif()
