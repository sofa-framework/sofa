cmake_minimum_required(VERSION 2.8)

project("SofaSimulationBGL")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(HEADER_FILES

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

set(SOURCE_FILES

	BglNode.cpp 
	BglMultiMappingElement.cpp 
	BglSimulation.cpp 
	BglGraphManager.cpp  
	GetObjectsVisitor.cpp
	)

if(SOFA_SMP)
	list(APPEND HEADER_FILES "SMPBglSimulation.h")
	list(APPEND SOURCE_FILES "SMPBglSimulation.cpp")
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaSimulationCommon)

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_DEFINITIONS "${GLOBAL_DEFINES};SOFA_BUILD_SIMULATION_BGL")
	
include(${SOFA_CMAKE_DIR}/post.cmake)

