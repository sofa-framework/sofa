cmake_minimum_required(VERSION 3.1)

project(SofaSimulationCore_simutest)

find_package(SofaComponentBase REQUIRED)

set(SOURCE_FILES
    DefaultAnimationLoop_test.cpp
    NodeContext_test.cpp
    )

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaComponentBase)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
