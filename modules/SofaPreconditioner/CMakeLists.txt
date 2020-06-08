cmake_minimum_required(VERSION 3.12)
project(SofaPreconditioner VERSION 1.0)

# Dependencies
find_package(SofaCommon REQUIRED) # SofaImplicitOdeSolver SofaSimpleFem
find_package(SofaGeneral REQUIRED) # SofaGeneralLinearSolver
sofa_find_package(SofaSparseSolver QUIET) # Soft dependency of PrecomputedWarpPreconditioner

set(SRC_ROOT src/SofaPreconditioner)

# Config
set(HEADER_FILES
    ${SRC_ROOT}/config.h.in
    )
set(SOURCE_FILES
    ${SRC_ROOT}/initPreconditioner.cpp
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

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaImplicitOdeSolver SofaGeneralLinearSolver SofaSimpleFem)
if(SofaSparseSolver_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC SofaSparseSolver)
endif()

sofa_generate_package(
    NAME SofaPreconditioner
    VERSION ${PROJECT_VERSION}
    TARGETS SofaPreconditioner
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "SofaPreconditioner"
    RELOCATABLE "plugins"
    )
