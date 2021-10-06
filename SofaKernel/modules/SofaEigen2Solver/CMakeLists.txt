cmake_minimum_required(VERSION 3.12)
project(SofaEigen2Solver LANGUAGES CXX)

# message(WARNING "SofaEigen2Solver has been removed, the package still exists but is empty. Eigen classes have moved to Sofa.LinearAlgebra and SVDLinearSolver to SofaDenseSolver.")

set(SOFAEIGEN2SOLVER_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAEIGEN2SOLVER_SRC}/config.h.in
    ${SOFAEIGEN2SOLVER_SRC}/initSofaEigen2Solver.h
)

set(SOURCE_FILES
    ${SOFAEIGEN2SOLVER_SRC}/initSofaEigen2Solver.cpp
)

sofa_find_package(Sofa.Helper REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Helper)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)
