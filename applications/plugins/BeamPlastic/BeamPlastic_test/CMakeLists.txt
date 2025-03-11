cmake_minimum_required(VERSION 3.12)

project(BeamPlastic_test VERSION 1.0)

find_package(Sofa.Testing REQUIRED)
find_package(Sofa.Component.StateContainer REQUIRED)

set(SOURCE_FILES
    BeamPlasticFEMForceField_test.cpp
    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} PUBLIC BeamPlastic) 
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Testing Sofa.Component.StateContainer) 

add_definitions("-DPLASTICBEAM_TEST_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\"")

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
