cmake_minimum_required(VERSION 2.8)

set(SOLUTION_NAME "SofaGui")
project(${SOLUTION_NAME})

include(${SOFA_CMAKE_DIR}/pre.cmake)

include("SofaGuiCommon.cmake")

if(GUI_USE_QTVIEWER)
	add_subdirectory("qt")
endif()

if(GUI_USE_GLUT)
	add_subdirectory("glut")
endif()

include("SofaGuiMain.cmake")

include(${SOFA_CMAKE_DIR}/post.cmake)
