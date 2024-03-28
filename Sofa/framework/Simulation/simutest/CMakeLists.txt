cmake_minimum_required(VERSION 3.12)

project(Sofa.Simulation_simutest)

set(SOURCE_FILES
    parallel_scenes_test.cpp
    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.Simulation.Core Sofa.Component)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
