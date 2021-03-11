cmake_minimum_required(VERSION 3.12)

project(SofaMiscMapping_test)

sofa_find_package(SofaMiscMapping REQUIRED)

set(HEADER_FILES)

set(SOURCE_FILES ../../empty.cpp)

list(APPEND SOURCE_FILES
    DistanceMapping_test.cpp
    SubsetMultiMapping_test.cpp
    SquareDistanceMapping_test.cpp
    SquareMapping_test.cpp
    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing SofaTest SofaMiscMapping)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
