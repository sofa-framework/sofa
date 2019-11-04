cmake_minimum_required(VERSION 3.1)

project(SofaSimulationCommon_test)

set(SOURCE_FILES
    LoadScene_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaHelper)
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../../SofaSimulationCore/SofaSimulationCore_test>")

add_test(NAME SofaSimulationCore_test COMMAND SofaSimulationCore_test)

