cmake_minimum_required(VERSION 3.12)
project(SofaGeneralImplicitOdeSolver LANGUAGES CXX)

set(SOFAGENERALIMPLICITODESOLVER_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAGENERALIMPLICITODESOLVER_SRC}/config.h.in
    ${SOFAGENERALIMPLICITODESOLVER_SRC}/initSofaGeneralImplicitODESolver.h
    )

set(SOURCE_FILES
    ${SOFAGENERALIMPLICITODESOLVER_SRC}/initSofaGeneralImplicitODESolver.cpp
    )

list(APPEND HEADER_FILES
    ${SOFAGENERALIMPLICITODESOLVER_SRC}/VariationalSymplecticSolver.h
    )
list(APPEND SOURCE_FILES
    ${SOFAGENERALIMPLICITODESOLVER_SRC}/VariationalSymplecticSolver.cpp
    )

find_package(SofaFramework REQUIRED) # SofaSimulationCore

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
cmake_dependent_option(SOFAGENERALIMPLICITODESOLVER_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFAGENERALIMPLICITODESOLVER_BUILD_TESTS)
    enable_testing()
    add_subdirectory(${PROJECT_NAME}_test)
endif()

