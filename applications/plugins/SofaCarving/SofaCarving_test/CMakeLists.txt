cmake_minimum_required(VERSION 3.12)

project(SofaCarving_test)

set(SOURCE_FILES
    SofaCarving_test.cpp
)

find_package(SofaComponentAll REQUIRED)

add_definitions("-DSOFACARVING_TEST_RESOURCES_DIR=\"${CMAKE_SOURCE_DIR}/share\"")
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest SofaCarving SofaSimulationGraph SofaComponentAll)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})

