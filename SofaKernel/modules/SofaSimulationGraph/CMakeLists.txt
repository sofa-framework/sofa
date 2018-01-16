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

if(SOFA_BUILD_TESTS)
    list(APPEND HEADER_FILES testing/BaseSimulationTest.h)
    list(APPEND SOURCE_FILES testing/BaseSimulationTest.cpp)
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationCommon)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SIMULATION_GRAPH")

sofa_install_targets(SofaSimulation ${PROJECT_NAME} ${PROJECT_NAME})

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/" DESTINATION "include/SofaSimulationGraph"
        FILES_MATCHING PATTERN "*.h" PATTERN "*.inl")
