# Find the metis headers and libraries
# Behavior is to first look for config files, such as the one installed by some package
# managers who provides their own cmake files.
# Most of them and official sources does not provide cmake finders, so if no config files
# were found, this tries to find the library by looking at headers / lib file.
#
# Defines:
#   Metis_FOUND : True if metis is found
#
# Provides both targets metis and metis::metis.
#   Target metis::metis is just an alias to metis.
# We chose to create an alias to provide a unified interface usable whatever the package manager
# was used to provide the library, as some package managers (such vcpkg) defines only short name
# for the target, whereas others (such as conan) defines a fully qualified name.

find_package(metis NO_MODULE)

if(TARGET metis)
  set(Metis_FOUND TRUE) # only metis_FOUND has been set
  add_library(metis::metis ALIAS metis)
else()

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

    if(NOT TARGET metis)
      add_library(metis INTERFACE IMPORTED)
      set_property(TARGET metis PROPERTY INTERFACE_LINK_LIBRARIES "${Metis_LIBRARIES}")
      set_property(TARGET metis PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${Metis_INCLUDE_DIR}")
    endif()
    add_library(metis::metis ALIAS metis)
  else()
  endif()
  mark_as_advanced(Metis_INCLUDE_DIR Metis_LIBRARY)
endif()
