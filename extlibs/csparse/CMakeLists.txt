cmake_minimum_required(VERSION 2.8)

project("csparse")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(HEADER_FILES

    csparse.h 
    ldl.h 
    UFconfig.h 
	)

set(SOURCE_FILES

    csparse.c 
    ldl.c 
	)

add_library(${PROJECT_NAME} STATIC ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILE_DEFINES "use_namespace")

include(${SOFA_CMAKE_DIR}/post.cmake)

