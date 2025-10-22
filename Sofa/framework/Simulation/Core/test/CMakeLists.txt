cmake_minimum_required(VERSION 3.22)

project(Sofa.Simulation.Core_test)

set(SOURCE_FILES
    Colors_test.cpp
    ParallelForEach_test.cpp
    RequiredPlugin_test.cpp
    SceneCheckRegistry_test.cpp
    Simulation_test.cpp
    ParallelSparseMatrixProduct_test.cpp
    TaskSchedulerFactory_test.cpp
    TaskSchedulerTestTasks.cpp
    TaskSchedulerTestTasks.h
    TaskSchedulerTests.cpp
    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.Simulation.Core Sofa.LinearAlgebra.Testing)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
