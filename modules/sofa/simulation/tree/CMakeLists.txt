cmake_minimum_required(VERSION 2.8)

project("SofaSimulationTree")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(HEADER_FILES

	tree.h
	ExportDotVisitor.h
	GNode.h
	GNodeVisitor.h
	GNodeMultiMappingElement.h
	TreeSimulation.h
	)

set(SOURCE_FILES

	ExportDotVisitor.cpp
	GNode.cpp
	GNodeVisitor.cpp
	GNodeMultiMappingElement.cpp
	TreeSimulation.cpp
	)

if(SOFA_SMP)
	list(APPEND HEADER_FILES "SMPSimulation.h")
	list(APPEND SOURCE_FILES "SMPSimulation.cpp")
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaSimulationCommon)

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_DEFINITIONS "${GLOBAL_DEFINES};SOFA_BUILD_SIMULATION_TREE")
	
include(${SOFA_CMAKE_DIR}/post.cmake)

