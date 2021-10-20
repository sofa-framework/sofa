cmake_minimum_required(VERSION 3.12)

project(SofaGeneralSimpleFem_test)

sofa_find_package(SofaGeneralSimpleFem REQUIRED)
sofa_find_package(SofaBaseMechanics REQUIRED)

set(SOURCE_FILES
    BeamFEMForceField_test.cpp
    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing SofaGeneralSimpleFem SofaBaseMechanics)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
