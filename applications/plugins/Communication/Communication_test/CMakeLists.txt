cmake_minimum_required(VERSION 3.1)

project(Communication_test)

set(SOURCE_FILES
    Communication_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest Communication)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
