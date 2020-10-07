cmake_minimum_required(VERSION 3.12)
project(SofaMiscSolver VERSION 1.0 LANGUAGES CXX)

find_package(SofaSimulation REQUIRED) # SofaSimulationTree

set(SOFAMISCSOLVER_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAMISCSOLVER_SRC}/config.h.in
    ${SOFAMISCSOLVER_SRC}/initMiscSolver.h
    ${SOFAMISCSOLVER_SRC}/DampVelocitySolver.h
    ${SOFAMISCSOLVER_SRC}/NewmarkImplicitSolver.h
)

set(SOURCE_FILES
    ${SOFAMISCSOLVER_SRC}/initMiscSolver.cpp
    ${SOFAMISCSOLVER_SRC}/DampVelocitySolver.cpp
    ${SOFAMISCSOLVER_SRC}/NewmarkImplicitSolver.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationTree)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
    )

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFAMISCSOLVER_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFAMISCSOLVER_BUILD_TESTS)
    enable_testing()
    add_subdirectory(SofaMiscSolver_test)
endif()
