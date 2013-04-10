cmake_minimum_required(VERSION 2.8)

project("RunSofa")

include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES

	Main.cpp
	)

if(GUI_USE_QTVIEWER)
	UseQt()
endif()
	
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

set(LINKER_DEPENDENCIES SofaGuiMain SofaComponentMain)
if(SIMULATION_GRAPH_DAG)
    list(APPEND LINKER_DEPENDENCIES SofaSimulationGraph)
endif()
include(${SOFA_CMAKE_DIR}/postProject.cmake)

