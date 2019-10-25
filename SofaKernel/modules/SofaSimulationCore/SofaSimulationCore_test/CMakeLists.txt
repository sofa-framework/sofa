cmake_minimum_required(VERSION 3.1)

project(SofaSimulationCore_test)

set(SOURCE_FILES
    TaskSchedulerTests.cpp
    TaskSchedulerTestTasks.h
    TaskSchedulerTestTasks.cpp
    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaHelper)

add_test(NAME SofaSimulationCore_test COMMAND SofaSimulationCore_test)
