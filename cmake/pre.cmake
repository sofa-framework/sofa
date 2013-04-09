cmake_minimum_required(VERSION 2.8)

if(NOT PROJECT_NAME STREQUAL "${SOLUTION_NAME}")
	message(STATUS "  > ${PROJECT_NAME} : Configuring Project")
else()
	message(STATUS "> ${PROJECT_NAME} : Configuring Solution\n")
endif()

# cmake modules path, for our FindXXX.cmake modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${SOFA_CMAKE_DIR})

#include(${CMAKE_CURRENT_LIST_DIR}/environment.cmake) #moved to root CMakeLists.txt
include(${CMAKE_CURRENT_LIST_DIR}/functions.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/options.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/pre-config.cmake)
