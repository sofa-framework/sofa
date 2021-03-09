cmake_minimum_required(VERSION 3.12)
project(SofaSimulationGraph LANGUAGES CXX)

set(SOFASIMULATIONGRAPH_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFASIMULATIONGRAPH_SRC}/config.h.in
    ${SOFASIMULATIONGRAPH_SRC}/init.h
    ${SOFASIMULATIONGRAPH_SRC}/initSofaSimulationGraph.h
    ${SOFASIMULATIONGRAPH_SRC}/graph.h
    ${SOFASIMULATIONGRAPH_SRC}/DAGNode.h
    ${SOFASIMULATIONGRAPH_SRC}/DAGNodeMultiMappingElement.h
    ${SOFASIMULATIONGRAPH_SRC}/DAGSimulation.h
    ${SOFASIMULATIONGRAPH_SRC}/SimpleApi.h
)

set(SOURCE_FILES
    ${SOFASIMULATIONGRAPH_SRC}/init.cpp
    ${SOFASIMULATIONGRAPH_SRC}/initSofaSimulationGraph.cpp
    ${SOFASIMULATIONGRAPH_SRC}/DAGNode.cpp
    ${SOFASIMULATIONGRAPH_SRC}/DAGNodeMultiMappingElement.cpp
    ${SOFASIMULATIONGRAPH_SRC}/DAGSimulation.cpp
    ${SOFASIMULATIONGRAPH_SRC}/SimpleApi.cpp
)

if(SOFA_BUILD_TESTS OR SOFA_BUILD_RELEASE_PACKAGE)
    list(APPEND HEADER_FILES ${SOFASIMULATIONGRAPH_SRC}/testing/BaseSimulationTest.h)
    list(APPEND SOURCE_FILES ${SOFASIMULATIONGRAPH_SRC}/testing/BaseSimulationTest.cpp)
endif()

sofa_find_package(SofaSimulationCommon REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationCommon)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFASIMULATIONGRAPH_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFASIMULATIONGRAPH_BUILD_TESTS)
    enable_testing()
    add_subdirectory(${PROJECT_NAME}_test)
endif()
