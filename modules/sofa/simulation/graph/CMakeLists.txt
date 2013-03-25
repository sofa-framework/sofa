cmake_minimum_required(VERSION 2.8)

project("SofaSimulationGraph")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(headers

	graph.h 
	DAGNode.h 
	DAGNodeMultiMappingElement.h 
	DAGSimulation.h 
	DAGSubGraphNode.h
	)

set(sources

	DAGNode.cpp 
	DAGNodeMultiMappingElement.cpp 
	DAGSimulation.cpp 
	DAGSubGraphNode.cpp
	)

add_library(${PROJECT_NAME} SHARED ${headers} ${sources})
target_link_libraries(${PROJECT_NAME} SofaSimulation)

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_DEFINITIONS "${GLOBAL_DEFINES};SOFA_BUILD_SIMULATION_GRAPH")
	
include(${SOFA_CMAKE_DIR}/post.cmake)

