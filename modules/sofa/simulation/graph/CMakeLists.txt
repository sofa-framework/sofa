include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

	graph.h 
	init.h 
	DAGNode.h 
	DAGNodeMultiMappingElement.h 
	DAGSimulation.h 
	)

set(SOURCE_FILES

	init.cpp
	DAGNode.cpp 
	DAGNodeMultiMappingElement.cpp 
	DAGSimulation.cpp 
	)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_SIMULATION_GRAPH")
AddLinkerDependencies(SofaSimulationCommon)

include(${SOFA_CMAKE_DIR}/postProject.cmake)

