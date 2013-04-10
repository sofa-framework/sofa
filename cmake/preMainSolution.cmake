cmake_minimum_required(VERSION 2.8)

set(GENERATED_FROM_MAIN_SOLUTION 1)

include(${CMAKE_CURRENT_LIST_DIR}/environment.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/functions.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/options.cmake)

unset(GLOBAL_DEPENDENCIES CACHE) # reset the dependency database (used to compute interdependencies)

if(FIRST_CONFIGURE_DONE)
	message(STATUS "--------------------------------------------")
	message(STATUS "-------- CONFIGURING SOFA FRAMEWORK --------")
	message(STATUS "--------------------------------------------")
	message(STATUS "")
	
	include(${CMAKE_CURRENT_LIST_DIR}/externals.cmake)
	include(${CMAKE_CURRENT_LIST_DIR}/buildFlags.cmake)
else()
	message("")
	message("Select your options and launch 'configure'")
	message("")
endif()