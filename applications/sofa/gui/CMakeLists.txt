set(SOLUTION_NAME "SofaGui")
project(${SOLUTION_NAME})

include(${SOFA_CMAKE_DIR}/preSolution.cmake)

RegisterDependencies("SofaGuiCommon" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaGuiCommon")
RegisterDependencies("SofaGuiMain" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaGuiMain")
RegisterDependencies("SofaGuiQt" OPTION "SOFA-LIB_GUI_QTVIEWER" COMPILE_DEFINITIONS "SOFA_GUI_QTVIEWER" PATH "${CMAKE_CURRENT_SOURCE_DIR}/qt")
RegisterDependencies("SofaGuiGlut" OPTION "SOFA-LIB_GUI_GLUT" COMPILE_DEFINITIONS "SOFA_GUI_GLUT" PATH "${CMAKE_CURRENT_SOURCE_DIR}/glut")

include(${SOFA_CMAKE_DIR}/postSolution.cmake)
