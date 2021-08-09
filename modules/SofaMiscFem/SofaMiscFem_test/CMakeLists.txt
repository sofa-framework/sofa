cmake_minimum_required(VERSION 3.12)
project(SofaMiscFem_test)

sofa_find_package(SofaMiscFem REQUIRED)

set(HEADER_FILES
    )
set(SOURCE_FILES
    # Unit test TODO (htalbot): improve the parameters and initialization
    TetrahedronHyperelasticityFEMForceField_params_test.cpp
    # Test of regression for hyperelasticity (MooneyRivlin)
    TetrahedronHyperelasticityFEMForceField_scene_test.cpp
    TriangleFEMForceField_test.cpp
    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
add_definitions("-DSOFAMISCFEM_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")
target_link_libraries(${PROJECT_NAME} Sofa.Testing SofaMiscFem SofaBase SceneCreator SofaGeneralDeformable SofaImplicitOdeSolver SofaEngine SofaGeneralEngine SofaBoundaryCondition SofaMiscMapping SofaMiscForceField)
add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
