cmake_minimum_required(VERSION 2.8)

if(NOT PROJECT_NAME STREQUAL "${SOLUTION_NAME}")
	include(${CMAKE_CURRENT_LIST_DIR}/post-config.cmake)
endif()

message("> ${PROJECT_NAME} : Done\n")