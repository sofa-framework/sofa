cmake_minimum_required(VERSION 3.22)
project(Sofa.SimpleApi_test)

set(SOURCE_FILES
    SimpleApi_test.cpp
    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.Simulation.Graph)

add_test(Name ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
