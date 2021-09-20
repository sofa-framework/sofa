cmake_minimum_required(VERSION 3.12)

project(SofaDenseSolver_test)

set(SOURCE_FILES
    NewMatMatrix_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} Sofa.Testing SofaDenseSolver SofaDefaultType.Testing)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
