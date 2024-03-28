cmake_minimum_required(VERSION 3.22)

project(Sofa.Simulation.Core_simutest)

set(SOURCE_FILES
    DefaultAnimationLoop_test.cpp
    NodeContext_test.cpp
    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.Simulation.Core Sofa.Component.SceneUtility)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
