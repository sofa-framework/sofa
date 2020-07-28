cmake_minimum_required(VERSION 3.12)
project(SofaSimulationGraph)

set(HEADER_FILES
    config.h.in
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

if(SOFA_BUILD_TESTS OR SOFA_BUILD_RELEASE_PACKAGE)
    list(APPEND HEADER_FILES testing/BaseSimulationTest.h)
    list(APPEND SOURCE_FILES testing/BaseSimulationTest.cpp)
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationCommon)

sofa_add_targets_to_package(
    PACKAGE_NAME SofaSimulation
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    )
