cmake_minimum_required(VERSION 3.12)
project(SofaPreconditioner)

set(SRC_ROOT src/SofaPreconditioner)

# Config
set(HEADER_FILES
    ${SRC_ROOT}/config.h.in
    ${SRC_ROOT}/initSofaPreconditioner.h
    )
set(SOURCE_FILES
    ${SRC_ROOT}/initSofaPreconditioner.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

#message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. Use Sofa.Component.LinearSolver.Iterative and Sofa.Component.LinearSolver.Preconditioner instead.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.LinearSolver.Preconditioner REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.LinearSolver.Preconditioner)

sofa_create_package_with_targets(
    PACKAGE_NAME SofaPreconditioner
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS SofaPreconditioner AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "SofaPreconditioner"
    RELOCATABLE "collections"
    )
