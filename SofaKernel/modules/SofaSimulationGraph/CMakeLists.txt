cmake_minimum_required(VERSION 3.1)
project(SofaSimulationGraph)

set(HEADER_FILES
    DAGNode.h
    DAGNodeMultiMappingElement.h
    DAGSimulation.h
    SimpleApi.h
    init.h
    graph.h
)

set(SOURCE_FILES
    DAGNode.cpp
    DAGNodeMultiMappingElement.cpp
    DAGSimulation.cpp
    SimpleApi.cpp
    init.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationCommon)

## Add to the list of include directory the one with the current files.
## In the src directory is a mimickin of the namespace structure so we always know where
## files are.
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}./>
    $<INSTALL_INTERFACE:./>
)

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SIMULATION_GRAPH")


sofa_install_targets(SofaSimulation ${PROJECT_NAME} ${PROJECT_NAME})

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/" DESTINATION "include/SofaSimulationGraph"
        FILES_MATCHING PATTERN "*.h" PATTERN "*.inl")
