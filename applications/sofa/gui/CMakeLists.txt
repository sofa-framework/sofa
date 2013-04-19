cmake_minimum_required(VERSION 2.8)

set(SOLUTION_NAME "SofaGui")
project(${SOLUTION_NAME})

include(${SOFA_CMAKE_DIR}/preSolution.cmake)

RegisterDependencies("SofaGuiCommon" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaGuiCommon")
RegisterDependencies("SofaGuiMain" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaGuiMain")

RegisterDependencies("SofaGuiQt" OPTION GUI_USE_QTVIEWER COMPILE_DEFINITIONS SOFA_GUI_QTVIEWER PATH "${CMAKE_CURRENT_SOURCE_DIR}/qt")
#if(GUI_USE_QTVIEWER)
#        add_subdirectory("qt")
#endif()

RegisterDependencies("SofaGuiGlut" OPTION GUI_USE_GLUT COMPILE_DEFINITIONS SOFA_GUI_GLUT PATH "${CMAKE_CURRENT_SOURCE_DIR}/glut")
#if(GUI_USE_GLUT)
#        add_subdirectory("glut")
#endif()

include(${SOFA_CMAKE_DIR}/postSolution.cmake)
