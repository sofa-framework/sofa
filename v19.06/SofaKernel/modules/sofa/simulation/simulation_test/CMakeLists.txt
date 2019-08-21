cmake_minimum_required(VERSION 3.1)

project(SofaSimulation_test)

set(SOURCE_FILES
    Node_test.h
    tree/GNode_test.cpp
    graph/DAG_test.cpp
    graph/Node_test.cpp
    graph/Simulation_test.cpp
    graph/SimpleApi_test.cpp
)

find_package(SofaTest REQUIRED)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
add_definitions("-DSOFASIMULATION_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")
target_link_libraries(${PROJECT_NAME} SofaGTestMain)
target_link_libraries(${PROJECT_NAME} SceneCreator SofaSimulationCommon SofaSimulationTree SofaSimulationGraph SofaComponentBase)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
