cmake_minimum_required(VERSION 3.12)
project(SofaEigen2Solver LANGUAGES CXX)

set(SOFAEIGEN2SOLVER_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAEIGEN2SOLVER_SRC}/config.h.in
    ${SOFAEIGEN2SOLVER_SRC}/initSofaEigen2Solver.h
    ${SOFAEIGEN2SOLVER_SRC}/SVDLinearSolver.h
)

set(SOURCE_FILES
    ${SOFAEIGEN2SOLVER_SRC}/initSofaEigen2Solver.cpp
    ${SOFAEIGEN2SOLVER_SRC}/SVDLinearSolver.cpp
)

sofa_find_package(SofaBaseLinearSolver REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseLinearSolver)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)
