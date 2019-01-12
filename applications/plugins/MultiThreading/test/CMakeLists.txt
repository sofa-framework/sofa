cmake_minimum_required(VERSION 3.1)

project(MultiThreading_test)

set ( HEADER_FILES
)
set(SOURCE_FILES
)

find_package(SofaTest REQUIRED)

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${HEADER_FILES})
target_link_libraries(${PROJECT_NAME} SofaTest SofaGTestMain SofaSimulationCore)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})