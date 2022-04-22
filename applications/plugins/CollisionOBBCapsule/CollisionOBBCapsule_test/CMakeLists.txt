cmake_minimum_required(VERSION 3.12)

project(CollisionOBBCapsule_test)

set(HEADER_FILES
    OBBCapsPrimitiveCreator.h
)

set(SOURCE_FILES
    OBB_test.cpp
)

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})
add_definitions("-DSOFAMISCCOLLISION_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Testing CollisionOBBCapsule)

get_target_property(SOFAMESHCOLLISIONTEST_INCLUDES SofaMeshCollision_test INCLUDE_DIRECTORIES)

target_include_directories(${PROJECT_NAME} PUBLIC ${SOFAMESHCOLLISIONTEST_INCLUDES})

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
