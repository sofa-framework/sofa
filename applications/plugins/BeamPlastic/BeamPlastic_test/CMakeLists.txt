cmake_minimum_required(VERSION 3.1)

project(BeamPlastic_test VERSION 1.0)

find_package(BeamPlastic REQUIRED)
find_package(Sofa.Testing REQUIRED)
find_package(SofaGTestMain REQUIRED)

set(SOURCE_FILES
    BeamPlasticFEMForceField_test.cpp
    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")

target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Testing SofaBaseMechanics BeamPlastic SofaCore SofaConstraint SofaBaseMechanics SofaUserInteraction SofaCommon) 

add_definitions("-DPLASTICBEAM_TEST_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\"")

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
