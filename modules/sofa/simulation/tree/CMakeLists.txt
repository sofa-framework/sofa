cmake_minimum_required(VERSION 2.8)

project("SofaSimulationTree")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(headers

	tree.h
	ExportDotVisitor.h
	GNode.h
	GNodeVisitor.h
	GNodeMultiMappingElement.h
	TreeSimulation.h
	)

set(sources

	ExportDotVisitor.cpp
	GNode.cpp
	GNodeVisitor.cpp
	GNodeMultiMappingElement.cpp
	TreeSimulation.cpp
	)

if(SOFA_SMP)
	list(APPEND headers "SMPSimulation.h")
	list(APPEND sources "SMPSimulation.cpp")
endif()

add_library(${PROJECT_NAME} SHARED ${headers} ${sources})
target_link_libraries(${PROJECT_NAME} SofaSimulation)

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_DEFINITIONS "${GLOBAL_DEFINES};SOFA_BUILD_SIMULATION_TREE")
	
include(${SOFA_CMAKE_DIR}/post.cmake)

