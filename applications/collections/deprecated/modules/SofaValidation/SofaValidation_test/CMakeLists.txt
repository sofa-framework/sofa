cmake_minimum_required(VERSION 3.22)

project(SofaValidation_test)

set(SOURCE_FILES
Monitor_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaValidation Sofa.Testing)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
