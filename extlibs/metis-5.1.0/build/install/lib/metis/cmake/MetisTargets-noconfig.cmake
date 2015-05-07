#----------------------------------------------------------------
# Generated CMake target import file for configuration "".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "metis" for configuration ""
set_property(TARGET metis APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(metis PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "C"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libmetis.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS metis )
list(APPEND _IMPORT_CHECK_FILES_FOR_metis "${_IMPORT_PREFIX}/lib/libmetis.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
