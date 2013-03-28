cmake_minimum_required(VERSION 2.8)

project("QGLViewer")

include(${SOFA_CMAKE_DIR}/pre.cmake)

# AHEM.... todo later :-)

message(FATAL_ERROR "QGLViewer project no yet converted to cmake build system !")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

include(${SOFA_CMAKE_DIR}/post.cmake)

