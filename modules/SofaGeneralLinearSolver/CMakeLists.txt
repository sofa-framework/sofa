cmake_minimum_required(VERSION 3.12)
project(SofaGeneralLinearSolver LANGUAGES CXX)

set(SOFAGENERALLINEARSOLVER_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAGENERALLINEARSOLVER_SRC}/config.h.in
    ${SOFAGENERALLINEARSOLVER_SRC}/initSofaGeneralLinearSolver.h
    )

set(SOURCE_FILES
    ${SOFAGENERALLINEARSOLVER_SRC}/initSofaGeneralLinearSolver.cpp
    )

list(APPEND HEADER_FILES
    ${SOFAGENERALLINEARSOLVER_SRC}/BTDLinearSolver.h
    ${SOFAGENERALLINEARSOLVER_SRC}/BTDLinearSolver.inl
    ${SOFAGENERALLINEARSOLVER_SRC}/CholeskySolver.h
    ${SOFAGENERALLINEARSOLVER_SRC}/CholeskySolver.inl
    ${SOFAGENERALLINEARSOLVER_SRC}/MinResLinearSolver.h
    ${SOFAGENERALLINEARSOLVER_SRC}/MinResLinearSolver.inl
    )
list(APPEND SOURCE_FILES
    ${SOFAGENERALLINEARSOLVER_SRC}/BTDLinearSolver.cpp
    ${SOFAGENERALLINEARSOLVER_SRC}/CholeskySolver.cpp
    ${SOFAGENERALLINEARSOLVER_SRC}/MinResLinearSolver.cpp
    )

find_package(SofaBase REQUIRED)
find_package(SofaSimulation REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationCommon SofaBaseLinearSolver)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
# cmake_dependent_option(SOFAGENERALLINEARSOLVER_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
# if(SOFAGENERALLINEARSOLVER_BUILD_TESTS)
#     enable_testing()
#     add_subdirectory(${PROJECT_NAME}_test)
# endif()

