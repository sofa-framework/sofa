cmake_minimum_required(VERSION 3.1)

project(SofaDistanceGrid_test)

set(SOURCE_FILES
    DistanceGrid_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest SofaDistanceGrid)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
