cmake_minimum_required(VERSION 3.12)

project(SofaMiscForceField_test)

sofa_find_package(SofaMiscForceField REQUIRED)
sofa_find_package(SofaTopologyMapping REQUIRED) # Needed by tests at runtime

set(SOURCE_FILES
    MeshMatrixMass_test.cpp
    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest SofaMiscForceField)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
