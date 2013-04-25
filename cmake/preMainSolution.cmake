cmake_minimum_required(VERSION 2.8)

set(GENERATED_FROM_MAIN_SOLUTION 1)

include(${CMAKE_CURRENT_LIST_DIR}/environment.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/functions.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/options.cmake)

if(PRECONFIGURE_DONE)
	message(STATUS "--------------------------------------------")
	message(STATUS "-------- CONFIGURING SOFA FRAMEWORK --------")
	message(STATUS "--------------------------------------------")
	message(STATUS "")
	
	include(${CMAKE_CURRENT_LIST_DIR}/externals.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/buildFlags.cmake)
else()
	message("")      
	message("SOFA framework pre-configuration complete")
	message("Select your options and launch 'configure' or re-run cmake")
	message("")
endif()
