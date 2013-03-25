cmake_minimum_required(VERSION 2.8)

if(NOT PROJECT_NAME STREQUAL "${SOLUTION_NAME}")
	message(STATUS "  > ${PROJECT_NAME} : Generating Project")
else()
	message(STATUS "> ${PROJECT_NAME} : Generating Solution\n")
endif()

include(${CMAKE_CURRENT_LIST_DIR}/environment.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/functions.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/options.cmake)
if(NOT PROJECT_NAME STREQUAL "${SOLUTION_NAME}")
	include(${CMAKE_CURRENT_LIST_DIR}/pre-config.cmake)
endif()