cmake_minimum_required(VERSION 3.12)
project(SofaExplicitOdeSolver LANGUAGES CXX)

set(SOFAEXPLICITEODESOLVER_SRC "src/${PROJECT_NAME}")

find_package(SofaFramework REQUIRED)

set(HEADER_FILES
    ${SOFAEXPLICITEODESOLVER_SRC}/config.h.in
    ${SOFAEXPLICITEODESOLVER_SRC}/initSofaExplicitOdeSolver.h
    ${SOFAEXPLICITEODESOLVER_SRC}/EulerSolver.h
)

set(SOURCE_FILES
    ${SOFAEXPLICITEODESOLVER_SRC}/initSofaExplicitOdeSolver.cpp
    ${SOFAEXPLICITEODESOLVER_SRC}/EulerSolver.cpp
)

find_package(SofaFramework REQUIRED) # SofaCore SofaSimulationCore

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaCore SofaSimulationCore)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFAEXPLICITEODESOLVER_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFAEXPLICITEODESOLVER_BUILD_TESTS)
    enable_testing()
    add_subdirectory(${PROJECT_NAME}_test)
endif()
