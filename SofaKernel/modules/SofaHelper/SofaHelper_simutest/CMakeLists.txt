cmake_minimum_required(VERSION 3.1)

project(SofaHelper_simutest)

find_package(SofaComponentBase REQUIRED)

set(SOURCE_FILES
    AdvancedTimer_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaComponentBase)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
