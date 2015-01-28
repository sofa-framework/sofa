# - Config file for the SofaPython package
# It defines the following variables
#  SofaPython_INCLUDE_DIRS - include directories for SofaPython
#  SofaPython_LIBRARIES    - libraries to link against


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was SofaPythonConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

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

set_and_check(SOFAPYTHON_INCLUDE_DIR  "${PACKAGE_PREFIX_DIR}/include")

check_required_components(SofaPython)

find_package(SofaGui REQUIRED)

set(SOFA_HAVE_PYTHON "1")

if( NOT TARGET SofaPython )
	include("${CMAKE_CURRENT_LIST_DIR}/SofaPythonTargets.cmake")
endif() 

set(SofaPython_LIBRARIES SofaPython )
set(SofaPython_INCLUDE_DIRS ${SOFAPYTHON_INCLUDE_DIR} ${SofaGui_INCLUDE_DIRS} )
