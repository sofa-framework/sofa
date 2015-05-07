# CMake package configuration file for the metis library.
# It defines the following variables:
# - Metis_INCLUDE_DIRS
# - Metis_LIBRARIES


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was MetisConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

set_and_check(METIS_INCLUDE_DIR "/storage/vault/sofa/branches/build-system/extlibs/metis-5.1.0")

if(NOT TARGET metis)
	include("${CMAKE_CURRENT_LIST_DIR}/MetisTargets.cmake")
endif()

check_required_components(newmat)

# Variables for compatibility with MODULE mode find_package.
set(Metis_LIBRARIES metis)
set(Metis_INCLUDE_DIRS ${METIS_INCLUDE_DIR})
