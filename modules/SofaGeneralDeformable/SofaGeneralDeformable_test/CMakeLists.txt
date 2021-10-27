cmake_minimum_required(VERSION 3.12)

project(SofaGeneralDeformable_test)

sofa_find_package(SofaGeneralDeformable REQUIRED)
sofa_find_package(SofaBaseMechanics REQUIRED)

set(SOURCE_FILES
    TriangularBendingSprings_test.cpp
    )

find_package(SofaGeneralDeformable REQUIRED)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing SofaGeneralDeformable SofaBaseMechanics)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
