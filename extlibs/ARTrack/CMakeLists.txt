cmake_minimum_required(VERSION 2.8)

project("ARTrack")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(HEADER_FILES

    dtracklib.h
 
	)

set(SOURCE_FILES

    mainTracker.cpp

	)

add_library(${PROJECT_NAME} STATIC ${HEADER_FILES} ${SOURCE_FILES})

# fix link errors when linked with shared libraries
if(${CMAKE_COMPILER_IS_GNUCC})
    set(COMPILER_FLAGS "-fPIC")
endif()

set(COMPILER_DEFINES "use_namespace")

include(${SOFA_CMAKE_DIR}/post.cmake)

