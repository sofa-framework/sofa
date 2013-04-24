cmake_minimum_required(VERSION 2.8)

include(${SOFA_CMAKE_DIR}/preProject.cmake)

##plugin external dependencies
find_package( SofaPimGUIQtLibs REQUIRED)
include_directories(${SOFAPIMSGUIQT_INCLUDE_DIRS})

set(HEADER_FILES
    
    QMouseOperations.h

	)

set(SOURCE_FILES

    QMouseOperations.cpp

	)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_PIMGUIQT")
set(LINKER_DEPENDENCIES SofaPimGui SofaComponentCollision SofaPim)

include(${SOFA_CMAKE_DIR}/postProject.cmake)