cmake_minimum_required(VERSION 2.8)

message(STATUS "  > ${PROJECT_NAME} : Configuring Project")

if(NOT GENERATED_FROM_SOLUTION)
	include(${CMAKE_CURRENT_LIST_DIR}/environment.cmake)
	include(${CMAKE_CURRENT_LIST_DIR}/functions.cmake)
	include(${CMAKE_CURRENT_LIST_DIR}/options.cmake)
	include(${CMAKE_CURRENT_LIST_DIR}/externals.cmake)
	include(${CMAKE_CURRENT_LIST_DIR}/buildFlags.cmake)
endif()

include(${CMAKE_CURRENT_LIST_DIR}/preBuildConfig.cmake)
