cmake_minimum_required(VERSION 3.12)
project(Sofa.Component.LinearSolver.Preconditioner LANGUAGES CXX)

set(SOFACOMPONENTLINEARSOLVERPRECONDITIONER_SOURCE_DIR "src/sofa/component/linearsolver/preconditioner")

set(HEADER_FILES
    ${SOFACOMPONENTLINEARSOLVERPRECONDITIONER_SOURCE_DIR}/config.h.in
    ${SOFACOMPONENTLINEARSOLVERPRECONDITIONER_SOURCE_DIR}/init.h
    ${SOFACOMPONENTLINEARSOLVERPRECONDITIONER_SOURCE_DIR}/BlockJacobiPreconditioner.h
    ${SOFACOMPONENTLINEARSOLVERPRECONDITIONER_SOURCE_DIR}/BlockJacobiPreconditioner.inl
    ${SOFACOMPONENTLINEARSOLVERPRECONDITIONER_SOURCE_DIR}/JacobiPreconditioner.h
    ${SOFACOMPONENTLINEARSOLVERPRECONDITIONER_SOURCE_DIR}/JacobiPreconditioner.inl
    ${SOFACOMPONENTLINEARSOLVERPRECONDITIONER_SOURCE_DIR}/PrecomputedWarpPreconditioner.h
    ${SOFACOMPONENTLINEARSOLVERPRECONDITIONER_SOURCE_DIR}/PrecomputedWarpPreconditioner.inl
    ${SOFACOMPONENTLINEARSOLVERPRECONDITIONER_SOURCE_DIR}/SSORPreconditioner.h
    ${SOFACOMPONENTLINEARSOLVERPRECONDITIONER_SOURCE_DIR}/SSORPreconditioner.inl
    ${SOFACOMPONENTLINEARSOLVERPRECONDITIONER_SOURCE_DIR}/WarpPreconditioner.h
    ${SOFACOMPONENTLINEARSOLVERPRECONDITIONER_SOURCE_DIR}/WarpPreconditioner.inl
    ${SOFACOMPONENTLINEARSOLVERPRECONDITIONER_SOURCE_DIR}/RotationMatrixSystem.h
    ${SOFACOMPONENTLINEARSOLVERPRECONDITIONER_SOURCE_DIR}/PrecomputedMatrixSystem.h
)

set(SOURCE_FILES
    ${SOFACOMPONENTLINEARSOLVERPRECONDITIONER_SOURCE_DIR}/init.cpp
    ${SOFACOMPONENTLINEARSOLVERPRECONDITIONER_SOURCE_DIR}/BlockJacobiPreconditioner.cpp
    ${SOFACOMPONENTLINEARSOLVERPRECONDITIONER_SOURCE_DIR}/JacobiPreconditioner.cpp
    ${SOFACOMPONENTLINEARSOLVERPRECONDITIONER_SOURCE_DIR}/PrecomputedWarpPreconditioner.cpp
    ${SOFACOMPONENTLINEARSOLVERPRECONDITIONER_SOURCE_DIR}/SSORPreconditioner.cpp
    ${SOFACOMPONENTLINEARSOLVERPRECONDITIONER_SOURCE_DIR}/WarpPreconditioner.cpp
    ${SOFACOMPONENTLINEARSOLVERPRECONDITIONER_SOURCE_DIR}/RotationMatrixSystem.cpp
    ${SOFACOMPONENTLINEARSOLVERPRECONDITIONER_SOURCE_DIR}/PrecomputedMatrixSystem.cpp
)

sofa_find_package(Sofa.Simulation.Core REQUIRED)
sofa_find_package(Sofa.Component.ODESolver.Backward REQUIRED)
sofa_find_package(Sofa.Component.LinearSolver.Iterative REQUIRED)
sofa_find_package(Sofa.Component.LinearSolver.Direct REQUIRED)

find_package(metis 5.1.0 EXACT QUIET)
if(NOT metis_FOUND AND SOFA_ALLOW_FETCH_DEPENDENCIES)
    message("Sofa.Component.LinearSolver.Direct: DEPENDENCY metis NOT FOUND. SOFA_ALLOW_FETCH_DEPENDENCIES is ON, fetching metis...")
    include(FetchContent)
    FetchContent_Declare(metis
            GIT_REPOSITORY https://github.com/sofa-framework/METIS
            GIT_TAG v5.1.0-ModernInstall
    )
    FetchContent_MakeAvailable(metis)
elseif (NOT metis_FOUND)
    message(FATAL_ERROR "Sofa.Component.LinearSolver.Direct: DEPENDENCY metis NOT FOUND. SOFA_ALLOW_FETCH_DEPENDENCIES is OFF and thus cannot be fetched. Install metis (version=5.1.0), or enable SOFA_ALLOW_FETCH_DEPENDENCIES to fix this issue.")
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${WRAPPER_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Simulation.Core Sofa.Component.ODESolver.Backward Sofa.Component.LinearSolver.Iterative Sofa.Component.LinearSolver.Direct)
target_link_libraries(${PROJECT_NAME} PRIVATE metis)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)
