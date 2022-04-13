cmake_minimum_required(VERSION 3.12)

project(Sofa.Component.Collision.Response.Mapper_test)

set(SOURCE_FILES
    BaryMapper_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.Component.Collision.Testing)
target_link_libraries(${PROJECT_NAME} Sofa.Component.Collision.Response.Mapper)
add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
