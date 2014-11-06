cmake_minimum_required(VERSION 2.8)
project(SofaSimulationTree)

set(SOFASIMULATIONTREE_HEADERS
	ExportDotVisitor.h
	GNode.h
	GNodeVisitor.h
	GNodeMultiMappingElement.h
	TreeSimulation.h
	)

set(SOFASIMULATIONTREE_SOURCES

	ExportDotVisitor.cpp
	GNode.cpp
	GNodeVisitor.cpp
	GNodeMultiMappingElement.cpp
	TreeSimulation.cpp
	)

if(SOFA-MISC_SMP)
	list(APPEND SOFASIMULATIONTREE_HEADERS "SMPSimulation.h")
	list(APPEND SOFASIMULATIONTREE_SOURCES "SMPSimulation.cpp")
endif()

add_library(${PROJECT_NAME} SHARED ${SOFASIMULATIONTREE_HEADERS} ${SOFASIMULATIONTREE_SOURCES})
target_link_libraries( ${PROJECT_NAME} SofaSimulationCommon )
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFASIMULATION_BUILD_INCLUDE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFASIMULATION_DEPENDENCY_INCLUDE_DIRS}>")

set( SOFASIMULATIONTREE_COMPILER_FLAGS "-DSOFA_BUILD_SIMULATION_TREE")

set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFASIMULATION_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFASIMULATIONTREE_COMPILER_FLAGS}"  )


install(TARGETS ${PROJECT_NAME}
        COMPONENT SofaSimulation_libraries
        EXPORT    SofaSimulationTargets
        RUNTIME DESTINATION "${SOFASIMULATION_INSTALL_BIN_DIR}"
        LIBRARY DESTINATION "${SOFASIMULATION_INSTALL_LIB_DIR}"
        ARCHIVE DESTINATION "${SOFASIMULATION_INSTALL_LIB_DIR}"
)

install(DIRECTORY ${SofaSimulationTree_SOURCE_DIR}/ DESTINATION "${SOFASIMULATION_INSTALL_INCLUDE_DIR}/sofa/simulation/tree"
        COMPONENT SofaSimulation_headers
        FILES_MATCHING 
            PATTERN "*.h"
            PATTERN "*.inl"
            PATTERN "*.hxx"
            PATTERN "*.hpp"
        )
