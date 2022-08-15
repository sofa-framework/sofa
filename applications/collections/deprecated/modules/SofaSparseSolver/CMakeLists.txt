cmake_minimum_required(VERSION 3.12)
project(SofaSparseSolver LANGUAGES CXX)

set(SRC_ROOT src/SofaSparseSolver)

# Sources
set(HEADER_FILES
    ${SRC_ROOT}/config.h.in
    ${SRC_ROOT}/initSofaSparseSolver.h
)
set(SOURCE_FILES
    ${SRC_ROOT}/initSofaSparseSolver.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

#message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. Use Sofa.Component.LinearSolver.Iterative and Sofa.Component.LinearSolver.Direct instead.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.LinearSolver.Direct REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.LinearSolver.Direct)

sofa_create_package_with_targets(
    PACKAGE_NAME SofaSparseSolver
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS SofaSparseSolver AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "SofaSparseSolver"
    RELOCATABLE "collections"
)
