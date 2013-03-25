cmake_minimum_required(VERSION 2.8)

project("tinyxml")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(headers

    tinystr.h
    tinyxml.h
	)

set(sources

    tinystr.cpp
    tinyxml.cpp
    tinyxmlerror.cpp
    tinyxmlparser.cpp
	)

add_library(${PROJECT_NAME} SHARED ${headers} ${sources})
#target_link_libraries(${PROJECT_NAME})

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_DEFINITIONS "${GLOBAL_DEFINES};SOFA_BUILD_TINYXML")
	
include(${SOFA_CMAKE_DIR}/post.cmake)

