cmake_minimum_required(VERSION 3.1)
project(Regression_test)

find_package(SofaFramework)

set(HEADER_FILES
    Regression_test.h
)

set(SOURCE_FILES
    Regression_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${HEADER_FILES})
add_definitions("-DSOFA_SRC_DIR=\"${CMAKE_SOURCE_DIR}\"")
target_link_libraries(${PROJECT_NAME} SofaTest SofaGTestMain)

