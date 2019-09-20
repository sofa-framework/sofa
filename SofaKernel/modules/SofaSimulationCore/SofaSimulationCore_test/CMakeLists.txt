cmake_minimum_required(VERSION 3.1)

project(SofaSimulationCore_test)

set(HEADER_FILES
    Node_test.h
    )

set(SOURCE_FILES
    DefaultAnimationLoop_test.cpp
    NodeContext_test.cpp
    TaskSchedulerTests.cpp
    TaskSchedulerTestTasks.h
    TaskSchedulerTestTasks.cpp
    )

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaHelper)

add_test(NAME SofaSimulationCore_test COMMAND SofaSimulationCore_test)
