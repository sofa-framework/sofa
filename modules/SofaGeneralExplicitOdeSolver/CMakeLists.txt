cmake_minimum_required(VERSION 3.12)
project(SofaGeneralExplicitOdeSolver LANGUAGES CXX)

find_package(SofaSimulationCommon REQUIRED)
set(SOFAGENERALEXPLICITODESOLVER_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAGENERALEXPLICITODESOLVER_SRC}/config.h.in
    ${SOFAGENERALEXPLICITODESOLVER_SRC}/initSofaGeneralExplicitODESolver.h
    )

set(SOURCE_FILES
    ${SOFAGENERALEXPLICITODESOLVER_SRC}/initSofaGeneralExplicitODESolver.cpp
    )

list(APPEND HEADER_FILES
    ${SOFAGENERALEXPLICITODESOLVER_SRC}/CentralDifferenceSolver.h
    ${SOFAGENERALEXPLICITODESOLVER_SRC}/RungeKutta2Solver.h
    ${SOFAGENERALEXPLICITODESOLVER_SRC}/RungeKutta4Solver.h
    )
list(APPEND SOURCE_FILES
    ${SOFAGENERALEXPLICITODESOLVER_SRC}/CentralDifferenceSolver.cpp
    ${SOFAGENERALEXPLICITODESOLVER_SRC}/RungeKutta2Solver.cpp
    ${SOFAGENERALEXPLICITODESOLVER_SRC}/RungeKutta4Solver.cpp
    )

find_package(SofaSimulation REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaCore SofaSimulationCommon)

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
cmake_dependent_option(SOFAGENERALEXPLICITODESOLVER_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFAGENERALEXPLICITODESOLVER_BUILD_TESTS)
    enable_testing()
    add_subdirectory(${PROJECT_NAME}_test)
endif()

