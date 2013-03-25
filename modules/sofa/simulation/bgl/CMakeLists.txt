cmake_minimum_required(VERSION 2.8)

project("SofaSimulationBGL")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(headers

	bgl.h 
	bfs_adapter.h 
	dfs_adapter.h 
	dfv_adapter.h 
	BglNode.h 
	BglMultiMappingElement.h 
	BglSimulation.h 
	BglGraphManager.h 
	BglGraphManager.inl 
	GetObjectsVisitor.h
	)

set(sources

	BglNode.cpp 
	BglMultiMappingElement.cpp 
	BglSimulation.cpp 
	BglGraphManager.cpp  
	GetObjectsVisitor.cpp
	)

if(SOFA_SMP)
	list(APPEND headers "SMPBglSimulation.h")
	list(APPEND sources "SMPBglSimulation.cpp")
endif()

add_library(${PROJECT_NAME} SHARED ${headers} ${sources})
target_link_libraries(${PROJECT_NAME} SofaSimulation)

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_DEFINITIONS "${GLOBAL_DEFINES};SOFA_BUILD_SIMULATION_BGL")
	
include(${SOFA_CMAKE_DIR}/post.cmake)

