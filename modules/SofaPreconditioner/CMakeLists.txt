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

# Sources
list(APPEND HEADER_FILES
    ${SRC_ROOT}/BlockJacobiPreconditioner.h
    ${SRC_ROOT}/BlockJacobiPreconditioner.inl
    ${SRC_ROOT}/JacobiPreconditioner.h
    ${SRC_ROOT}/JacobiPreconditioner.inl
    ${SRC_ROOT}/PrecomputedWarpPreconditioner.h
    ${SRC_ROOT}/PrecomputedWarpPreconditioner.inl
    ${SRC_ROOT}/SSORPreconditioner.h
    ${SRC_ROOT}/SSORPreconditioner.inl
    ${SRC_ROOT}/ShewchukPCGLinearSolver.h
    ${SRC_ROOT}/WarpPreconditioner.h
    ${SRC_ROOT}/WarpPreconditioner.inl
    )
list(APPEND SOURCE_FILES
    ${SRC_ROOT}/BlockJacobiPreconditioner.cpp
    ${SRC_ROOT}/JacobiPreconditioner.cpp
    ${SRC_ROOT}/PrecomputedWarpPreconditioner.cpp
    ${SRC_ROOT}/SSORPreconditioner.cpp
    ${SRC_ROOT}/ShewchukPCGLinearSolver.cpp
    ${SRC_ROOT}/WarpPreconditioner.cpp
    )

# Dependencies
sofa_find_package(SofaImplicitOdeSolver REQUIRED)
sofa_find_package(SofaSimpleFem REQUIRED)
sofa_find_package(SofaGeneralLinearSolver REQUIRED)
sofa_find_package(SofaSparseSolver QUIET) # Soft dependency of PrecomputedWarpPreconditioner

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaImplicitOdeSolver SofaGeneralLinearSolver SofaSimpleFem)
if(SofaSparseSolver_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC SofaSparseSolver)
endif()

sofa_create_package_with_targets(
    PACKAGE_NAME SofaPreconditioner
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS SofaPreconditioner AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "SofaPreconditioner"
    RELOCATABLE "plugins"
    )
