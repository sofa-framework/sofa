cmake_minimum_required(VERSION 3.12)
project(Sofa.Simulation.Graph LANGUAGES CXX)

set(SOFASIMULATIONGRAPH_SRC "src/sofa/simulation/graph")

set(HEADER_FILES
    ${SOFASIMULATIONGRAPH_SRC}/config.h.in
    ${SOFASIMULATIONGRAPH_SRC}/init.h
    ${SOFASIMULATIONGRAPH_SRC}/initSofaSimulationGraph.h
    ${SOFASIMULATIONGRAPH_SRC}/DAGNode.h
    ${SOFASIMULATIONGRAPH_SRC}/DAGSimulation.h
)

set(SOURCE_FILES
    ${SOFASIMULATIONGRAPH_SRC}/init.cpp
    ${SOFASIMULATIONGRAPH_SRC}/initSofaSimulationGraph.cpp
    ${SOFASIMULATIONGRAPH_SRC}/DAGNode.cpp
    ${SOFASIMULATIONGRAPH_SRC}/DAGSimulation.cpp
)

sofa_find_package(Sofa.Simulation.Common REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Simulation.Common)

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER Sofa.Framework) # IDE folder

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFA_SIMULATION_GRAPH_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFA_SIMULATION_GRAPH_BUILD_TESTS)
    add_subdirectory(test)
endif()
