include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

	tree.h
	ExportDotVisitor.h
	GNode.h
	GNodeVisitor.h
	GNodeMultiMappingElement.h
	TreeSimulation.h
	)

set(SOURCE_FILES

	tree.cpp
	ExportDotVisitor.cpp
	GNode.cpp
	GNodeVisitor.cpp
	GNodeMultiMappingElement.cpp
	TreeSimulation.cpp
	)

if(SOFA-MISC_SMP)
	list(APPEND HEADER_FILES "SMPSimulation.h")
	list(APPEND SOURCE_FILES "SMPSimulation.cpp")
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_SIMULATION_TREE")
AddLinkerDependencies(SofaSimulationCommon)

include(${SOFA_CMAKE_DIR}/postProject.cmake)

