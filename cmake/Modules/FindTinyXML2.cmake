# Find the tinyxml2 headers and libraries
# Behavior is to first look for config files, such as the one installed by some package
# managers who provides their own cmake files. If no config files
# were found, this tries to find the library by looking at headers / lib file.
#
# Defines:
#   TinyXML2_FOUND : True if tinyxml2 is found
#
# Provides target tinyxml2::tinyxml2.
find_package(tinyxml2 NO_MODULE QUIET)

if(TARGET tinyxml2::tinyxml2)
  set(TinyXML2_FOUND TRUE) # only tinyxml2_FOUND has been set
else()

  if(NOT TinyXML2_INCLUDE_DIR)
    find_path(TinyXML2_INCLUDE_DIR
      NAMES tinyxml2.h
      PATH_SUFFIXES include
    )
  endif()

  if(NOT TinyXML2_LIBRARY)
  find_library(TinyXML2_LIBRARY
    NAMES tinyxml2
    PATH_SUFFIXES lib
  )
  endif()

  if(TinyXML2_INCLUDE_DIR AND TinyXML2_LIBRARY)
    set(TinyXML2_FOUND TRUE)
  endif()

  if(TinyXML2_FOUND)
    set(TinyXML2_LIBRARIES ${TinyXML2_LIBRARY})
    set(TinyXML2_INCLUDE_DIRS ${TinyXML2_INCLUDE_DIR})

    if(NOT TARGET tinyxml2::tinyxml2)
      add_library(tinyxml2::tinyxml2 INTERFACE IMPORTED)
      set_property(TARGET tinyxml2::tinyxml2 PROPERTY INTERFACE_LINK_LIBRARIES "${TinyXML2_LIBRARIES}")
      set_property(TARGET tinyxml2::tinyxml2 PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${TinyXML2_INCLUDE_DIR}")
    endif()
  else()
  endif()
  mark_as_advanced(TinyXML2_INCLUDE_DIR TinyXML2_LIBRARY)
endif()
