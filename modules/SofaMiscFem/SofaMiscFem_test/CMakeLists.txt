cmake_minimum_required(VERSION 3.1)
project(SofaMiscFem_test)

find_package(Metis QUIET)


set(SOURCE_FILES

    # Unit test TODO (htalbot): improve the parameters and initialization
    TetrahedronHyperelasticityFEMForceField_params_test.cpp
    
    )

if(Metis_FOUND)
    # Test of regression for hyperelasticity (MooneyRivlin)
    list(APPEND SOURCE_FILES "TetrahedronHyperelasticityFEMForceField_scene_test.cpp")
endif()

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
add_definitions("-DSOFAMISCFEM_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest)
add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})