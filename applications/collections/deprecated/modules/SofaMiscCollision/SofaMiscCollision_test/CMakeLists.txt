cmake_minimum_required(VERSION 3.12)
project(SofaMiscCollision_test)

set(HEADER_FILES
)

set(SOURCE_FILES
    DefaultCollisionGroupManager_test.cpp
)

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})
add_definitions("-DSOFAMISCCOLLISION_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Testing Sofa.Component.Collision.Testing SofaMiscCollision
        SofaBase SofaGeneralDeformable SofaImplicitOdeSolver SofaTopologyMapping SofaLoader)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
