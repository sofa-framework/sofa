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

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Metis REQUIRED_VARS Metis_LIBRARY Metis_INCLUDE_DIR HANDLE_COMPONENTS)

if(Metis_FOUND)
  set(Metis_LIBRARIES ${Metis_LIBRARY})
  set(Metis_INCLUDE_DIRS ${Metis_INCLUDE_DIR})

  if(NOT TARGET Metis::Metis)
    add_library(Metis::Metis INTERFACE IMPORTED)
    set_property(TARGET Metis::Metis PROPERTY INTERFACE_LINK_LIBRARIES "${Metis_LIBRARIES}")
    set_property(TARGET Metis::Metis PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${Metis_INCLUDE_DIR}")
  endif()
endif()
mark_as_advanced(Metis_INCLUDE_DIR Metis_LIBRARY)
