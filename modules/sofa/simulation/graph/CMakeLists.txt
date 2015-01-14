cmake_minimum_required(VERSION 2.8)
project(SofaSimulationGraph)

set(SOFASIMULATION_HEADERS
	DAGNode.h 
	DAGNodeMultiMappingElement.h 
	DAGSimulation.h 
	)

set(SOFASIMULATION_SOURCES

	DAGNode.cpp 
	DAGNodeMultiMappingElement.cpp 
	DAGSimulation.cpp 
	)

add_library(${PROJECT_NAME} SHARED ${SOFASIMULATION_HEADERS} ${SOFASIMULATION_SOURCES})
target_link_libraries( ${PROJECT_NAME} PUBLIC SofaSimulationCommon )
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFASIMULATION_BUILD_INCLUDE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFASIMULATION_DEPENDENCY_INCLUDE_DIRS}>")

set( SOFASIMULATIONGRAPH_COMPILER_FLAGS "-DSOFA_BUILD_SIMULATION_GRAPH")

set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFASIMULATION_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS ${SOFASIMULATIONGRAPH_COMPILER_FLAGS}  )

install(TARGETS ${PROJECT_NAME}
        COMPONENT SofaSimulation_libraries
        EXPORT  SofaSimulationTargets
        RUNTIME DESTINATION "${SOFASIMULATION_INSTALL_BIN_DIR}"
        LIBRARY DESTINATION "${SOFASIMULATION_INSTALL_LIB_DIR}"
        ARCHIVE DESTINATION "${SOFASIMULATION_INSTALL_LIB_DIR}"
)

install(DIRECTORY ${SofaSimulationGraph_SOURCE_DIR}/ DESTINATION "${SOFASIMULATION_INSTALL_INCLUDE_DIR}/sofa/simulation/graph"
        COMPONENT SofaSimulation_headers
        FILES_MATCHING 
            PATTERN "*.h"
            PATTERN "*.inl"
            PATTERN "*.hxx"
            PATTERN "*.hpp"
        )

