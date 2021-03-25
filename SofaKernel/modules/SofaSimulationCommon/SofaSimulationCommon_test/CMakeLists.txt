cmake_minimum_required(VERSION 3.12)

project(SofaSimulationCommon_test)

set(SOURCE_FILES
    LoadScene_test.cpp
)

add_definitions("-DSOFASIMULATION_TEST_SCENES_DIR=\"${CMAKE_SOURCE_DIR}/examples/Components/constraint\"")
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest SofaHelper SofaComponentAll)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})

