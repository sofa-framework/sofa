cmake_minimum_required(VERSION 2.8)

if(NOT PROJECT_NAME STREQUAL "${SOLUTION_NAME}")
	include(${CMAKE_CURRENT_LIST_DIR}/post-config.cmake)
endif()

if(NOT PROJECT_NAME STREQUAL "${SOLUTION_NAME}")
	message(STATUS "  > ${PROJECT_NAME} : Done\n")
else()
	message(STATUS "> ${PROJECT_NAME} : Done\n")
endif()

