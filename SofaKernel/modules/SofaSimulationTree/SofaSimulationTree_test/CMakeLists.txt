cmake_minimum_required(VERSION 3.1)

project(SofaSimulationTree_test)

set(SOURCE_FILES
    GNode_test.cpp
)

find_package(SofaTest REQUIRED)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain)
target_link_libraries(${PROJECT_NAME} SceneCreator SofaSimulationCommon SofaSimulationTree SofaSimulationGraph SofaBase)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
