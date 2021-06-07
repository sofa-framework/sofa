cmake_minimum_required(VERSION 3.12)

project(SofaMiscCollision_test)

find_package(SofaMiscCollision REQUIRED)

set(SOURCE_FILES
    DefaultCollisionGroupManager_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
add_definitions("-DSOFAMISCCOLLISION_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Testing SofaMiscCollision
        SofaBase SofaGeneralDeformable SofaImplicitOdeSolver SofaMeshCollision SofaTopologyMapping SofaLoader)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
