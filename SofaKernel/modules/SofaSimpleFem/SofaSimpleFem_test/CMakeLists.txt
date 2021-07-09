cmake_minimum_required(VERSION 3.12)

project(SofaSimpleFem_test)

set(HEADER_FILES
    ForceFieldTestCreation.h
)

set(SOURCE_FILES
    HexahedronFEMForceField_test.cpp
    TetrahedronFEMForceField_test.cpp
    TetrahedronDiffusionFEMForceField_test.cpp
)

sofa_find_package(SofaSimpleFem REQUIRED)

add_definitions("-DSOFASIMPLEFEM_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")
add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing SofaSimpleFem SofaBase SofaEngine SofaMeshCollision SofaImplicitOdeSolver SceneCreator)

# allow use ForceFieldTestCreation to other tests
target_include_directories(${PROJECT_NAME} PUBLIC ..)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
