cmake_minimum_required(VERSION 2.8)

project("SofaPimGui")

include(${SOFA_CMAKE_DIR}/preProject.cmake)

##plugin external dependencies
#find_package( SofaPimGUILibLibs REQUIRED)
#include_directories(${SOFAPIMSGUILIB_INCLUDE_DIRS})

set(HEADER_FILES
    
    ../MouseOperations.h

	)

set(SOURCE_FILES

    ../MouseOperations.cpp

	)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_PIMGUI")
set(LINKER_DEPENDENCIES SofaGui SofaComponentCollusion SofaPim)

include(${SOFA_CMAKE_DIR}/postProject.cmake)