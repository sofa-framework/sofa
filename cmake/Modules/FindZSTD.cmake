# Find the zstd headers and libraries
# Behavior is to first look for config files, such as the one installed by some package
# managers who provides their own cmake files. If no config files
# were found, this tries to find the library by looking at headers / lib file.
#
# Defines:
#   ZSTD_FOUND : True if ZSTD is found
#
# Provides target ZSTD::ZSTD.
# TODO: keep CONFIG mode ?
#find_package(zstd NO_MODULE QUIET)

if(TARGET ZSTD::ZSTD)
  set(ZSTD_FOUND TRUE) # only ZSTD_FOUND has been set
else()

  if(NOT ZSTD_INCLUDE_DIR)
    find_path(ZSTD_INCLUDE_DIR
      NAMES zstd.h
      PATH_SUFFIXES include
    )
  endif()

  if(NOT ZSTD_LIBRARY)
  find_library(ZSTD_LIBRARY
    NAMES zstd
    PATH_SUFFIXES lib
  )
  endif()

  if(WIN32)
    find_file(ZSTD_DLL
      NAMES zstd.dll
      PATH_SUFFIXES bin
    )
  endif()

  if(ZSTD_INCLUDE_DIR AND ZSTD_LIBRARY AND (NOT WIN32 OR ZSTD_DLL))
    set(ZSTD_FOUND TRUE)
  else()
    if(ZSTD_FIND_REQUIRED)
      message(FATAL_ERROR "Cannot find ZSTD")
    endif()
  endif()

  if(ZSTD_FOUND)
    set(ZSTD_LIBRARIES ${ZSTD_LIBRARY})
    set(ZSTD_INCLUDE_DIRS ${ZSTD_INCLUDE_DIR})

    if(NOT TARGET ZSTD::ZSTD)
      add_library(ZSTD::ZSTD SHARED IMPORTED)
      if(WIN32)
        set_property(TARGET ZSTD::ZSTD PROPERTY IMPORTED_LOCATION "${ZSTD_DLL}")
        set_property(TARGET ZSTD::ZSTD PROPERTY IMPORTED_IMPLIB "${ZSTD_LIBRARIES}")
      else()
        set_property(TARGET ZSTD::ZSTD PROPERTY IMPORTED_LOCATION "${ZSTD_LIBRARIES}")
      endif()
      set_property(TARGET ZSTD::ZSTD PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${ZSTD_INCLUDE_DIR}")
    endif()
  else()
  endif()
  mark_as_advanced(ZSTD_INCLUDE_DIR ZSTD_LIBRARY)
endif()
