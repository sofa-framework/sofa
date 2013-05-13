include(${SOFA_CMAKE_DIR}/preProject.cmake)

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

if(SOFA-MISC_HAVE_SMP)
	list(APPEND HEADER_FILES "SMPBglSimulation.h")
	list(APPEND SOURCE_FILES "SMPBglSimulation.cpp")
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_SIMULATION_BGL")
set(LINKER_DEPENDENCIES SofaSimulationCommon)

include(${SOFA_CMAKE_DIR}/postProject.cmake)

