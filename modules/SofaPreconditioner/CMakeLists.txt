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
    )
list(APPEND SOURCE_FILES
    )

# Dependencies
sofa_find_package(SofaImplicitOdeSolver REQUIRED)
sofa_find_package(SofaGeneralLinearSolver REQUIRED)
sofa_find_package(SofaSparseSolver QUIET) # Soft dependency of PrecomputedWarpPreconditioner

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaImplicitOdeSolver SofaGeneralLinearSolver)
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
