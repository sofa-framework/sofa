cmake_minimum_required(VERSION 3.12)
project(SofaEigen2Solver LANGUAGES CXX)

set(SOFAEIGEN2SOLVER_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAEIGEN2SOLVER_SRC}/config.h.in
    ${SOFAEIGEN2SOLVER_SRC}/initSofaEigen2Solver.h
)

set(SOURCE_FILES
    ${SOFAEIGEN2SOLVER_SRC}/initSofaEigen2Solver.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

#message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. Use Sofa.LinearAlgebra (Eigen classes) and Sofa.Component.LinearSolver.Direct (SVDLinearSolver) instead.")

# forward to the new sofang module
sofa_find_package(Sofa.LinearAlgebra REQUIRED)
sofa_find_package(Sofa.Component.LinearSolver.Direct REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.LinearAlgebra)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.LinearSolver.Direct)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "collections"
)
