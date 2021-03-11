cmake_minimum_required(VERSION 3.12)

project(SofaCarving_test)

set(SOURCE_FILES
    SofaCarving_test.cpp
)

find_package(SofaGTestMain REQUIRED)
find_package(SofaCarving REQUIRED)
find_package(SofaSimulationGraph REQUIRED)
find_package(SofaBaseUtils REQUIRED)
find_package(SofaBaseLinearSolver REQUIRED)

add_definitions("-DSOFACARVING_TEST_RESOURCES_DIR=\"${CMAKE_SOURCE_DIR}/share\"")
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} Sofa.Testing SofaTest SofaCarving SofaSimulationGraph SofaBaseUtils SofaBaseLinearSolver)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})

