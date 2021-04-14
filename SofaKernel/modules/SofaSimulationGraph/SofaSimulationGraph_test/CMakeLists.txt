cmake_minimum_required(VERSION 3.12)
project(SofaSimulationGraph_test)

set(HEADER_FILES
	Node_test.h
)

set(SOURCE_FILES
    DAG_test.cpp
    DAGNode_test.cpp
    MutationListener_test.cpp
    Node_test.cpp
    SimpleApi_test.cpp
    Simulation_test.cpp
    Link_test.cpp
    )

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing SofaTest)

add_test(Name ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
