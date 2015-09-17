cmake_minimum_required(VERSION 2.8.12)
project(SofaSimulationGraph)

set(HEADER_FILES
    DAGNode.h
    DAGNodeMultiMappingElement.h
    DAGSimulation.h
    init.h
)

set(SOURCE_FILES
    DAGNode.cpp
    DAGNodeMultiMappingElement.cpp
    DAGSimulation.cpp
    init.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationCommon)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SIMULATION_GRAPH")

sofa_install_targets(SofaSimulation ${PROJECT_NAME} ${PROJECT_NAME})

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/" DESTINATION "include/sofa/simulation/graph"
        FILES_MATCHING PATTERN "*.h" PATTERN "*.inl")
