cmake_minimum_required(VERSION 2.6)

project(Zinri)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

file(
	GLOB_RECURSE
	source_files
	${CMAKE_CURRENT_SOURCE_DIR}/*.c
)

add_library( Zinri ${source_files} )
