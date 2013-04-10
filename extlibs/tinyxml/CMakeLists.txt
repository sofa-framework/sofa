cmake_minimum_required(VERSION 2.8)

project("tinyxml")

include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    tinystr.h
    tinyxml.h
	)

set(SOURCE_FILES

    tinystr.cpp
    tinyxml.cpp
    tinyxmlerror.cpp
    tinyxmlparser.cpp
	)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_TINYXML")

include(${SOFA_CMAKE_DIR}/postProject.cmake)

