cmake_minimum_required(VERSION 3.12)

project(SofaMiscCollision_test)

find_package(SofaMiscCollision REQUIRED)

set(HEADER_FILES
    OBBCapsPrimitiveCreator.h
)

set(SOURCE_FILES
    DefaultCollisionGroupManager_test.cpp
    OBB_test.cpp
)

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})
add_definitions("-DSOFAMISCCOLLISION_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Testing SofaMiscCollision
        SofaBase SofaGeneralDeformable SofaImplicitOdeSolver SofaMeshCollision SofaTopologyMapping SofaLoader)

get_target_property(SOFAMESHCOLLISIONTEST_INCLUDES SofaMeshCollision_test INCLUDE_DIRECTORIES)

target_include_directories(${PROJECT_NAME} PUBLIC ${SOFAMESHCOLLISIONTEST_INCLUDES})
target_include_directories(${PROJECT_NAME} PUBLIC ..)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
