cmake_minimum_required(VERSION 2.8.12)

project(SofaMiscMapping_test)

set(SOURCE_FILES
    ProjectionToTargetLineMapping_test.cpp
    ProjectionToTargetPlaneMapping_test.cpp
    SubsetMultiMapping_test.cpp)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
