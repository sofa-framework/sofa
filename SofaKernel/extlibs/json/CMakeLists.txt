cmake_minimum_required(VERSION 3.1)

set(JSON_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		COMPONENT json_headers
		DESTINATION include
		PATTERN "*.in" EXCLUDE
		PATTERN "*.txt" EXCLUDE
		PATTERN "*.cpp" EXCLUDE)


#cmake_minimum_required(VERSION 3.1)
#project(json C CXX)

#set(HEADER_FILES
#    nlohmann_json.h
#    json.h
#)

#add_library(${PROJECT_NAME} INTERFACE)
#target_include_directories(${PROJECT_NAME} INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")

#include(../../SofaFramework/SofaMacros.cmake)
#sofa_create_package(json 0 ${PROJECT_NAME} "")
