cmake_minimum_required(VERSION 2.8)

project("SofaSimulationGraph")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(HEADER_FILES

	graph.h 
	DAGNode.h 
	DAGNodeMultiMappingElement.h 
	DAGSimulation.h 
	DAGSubGraphNode.h
	)

set(SOURCE_FILES

	DAGNode.cpp 
	DAGNodeMultiMappingElement.cpp 
	DAGSimulation.cpp 
	DAGSubGraphNode.cpp
	)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILE_DEFINES "SOFA_BUILD_SIMULATION_GRAPH")
set(LINK_DEPENDENCIES SofaSimulationCommon)

include(${SOFA_CMAKE_DIR}/post.cmake)

