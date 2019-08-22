cmake_minimum_required(VERSION 3.1)

project(SofaSimpleFem_test)

set(SOURCE_FILES
    HexahedronFEMForceField_test.cpp
    TetrahedronFEMForceField_test.cpp
    TetrahedronDiffusionFEMForceField_test.cpp
)

add_definitions("-DSOFASIMPLEFEM_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
