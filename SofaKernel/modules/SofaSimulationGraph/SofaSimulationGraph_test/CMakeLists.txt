cmake_minimum_required(VERSION 3.1)
project(SofaSimulationGraph_test)

find_package(SofaGTestMain REQUIRED)
find_package(SofaTest REQUIRED)

set(SOURCE_FILES
    DAG_test.cpp
    DAGNode_test.cpp
    MutationListener_test.cpp
    Node_test.cpp
    SimpleApi_test.cpp
    Simulation_test.cpp
    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest)

add_test(Name ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
