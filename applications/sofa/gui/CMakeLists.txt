cmake_minimum_required(VERSION 2.8)

set(SOLUTION_NAME "SofaGUI")
project(${SOLUTION_NAME})

include(${SOFA_CMAKE_DIR}/pre.cmake)

include("SofaGUICommon.cmake")
include("SofaGUIMain.cmake")

add_subdirectory("qt")
#add_subdirectory("glut")

include(${SOFA_CMAKE_DIR}/post.cmake)
