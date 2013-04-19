cmake_minimum_required(VERSION 2.8)

set(GENERATED_FROM_SOLUTION 1)

#message(STATUS "> ${PROJECT_NAME} : Configuring Solution\n")

if(NOT GENERATED_FROM_MAIN_SOLUTION)
	include(${CMAKE_CURRENT_LIST_DIR}/environment.cmake)
	include(${CMAKE_CURRENT_LIST_DIR}/functions.cmake)
	include(${CMAKE_CURRENT_LIST_DIR}/options.cmake)
	include(${CMAKE_CURRENT_LIST_DIR}/externals.cmake)
	include(${CMAKE_CURRENT_LIST_DIR}/buildFlags.cmake)
endif()