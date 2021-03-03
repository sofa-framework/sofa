cmake_minimum_required(VERSION 3.12)
project(SofaSparseSolver LANGUAGES CXX)

# Dependencies
sofa_find_package(SofaBase REQUIRED)
sofa_find_package(SofaImplicitOdeSolver REQUIRED) 
sofa_find_package(SofaSimpleFem REQUIRED)
sofa_find_package(SofaGeneralLinearSolver REQUIRED)

add_subdirectory(extlibs/csparse)
sofa_set_01(SOFASPARSESOLVER_HAVE_CSPARSE VALUE TRUE)

sofa_find_package(metis QUIET) # Unix users can have an installed version of metis
if(NOT metis_FOUND)
    message(STATUS "SofaSparseSolver: using built-in metis library")
    add_subdirectory(extlibs/metis-5.1.0)
endif()
sofa_set_01(SOFASPARSESOLVER_HAVE_METIS VALUE TRUE)

set(SRC_ROOT src/SofaSparseSolver)

# Sources
set(HEADER_FILES
    ${SRC_ROOT}/config.h.in
    ${SRC_ROOT}/PrecomputedLinearSolver.h
    ${SRC_ROOT}/PrecomputedLinearSolver.inl
    ${SRC_ROOT}/SparseLDLSolver.h
    ${SRC_ROOT}/SparseLDLSolver.inl
    ${SRC_ROOT}/SparseLDLSolverImpl.h
    ${SRC_ROOT}/SparseCholeskySolver.h
    ${SRC_ROOT}/SparseLUSolver.h
    ${SRC_ROOT}/SparseLUSolver.inl
    )
set(SOURCE_FILES
    ${SRC_ROOT}/initSparseSolver.cpp
    ${SRC_ROOT}/PrecomputedLinearSolver.cpp
    ${SRC_ROOT}/SparseLDLSolver.cpp
    ${SRC_ROOT}/SparseCholeskySolver.cpp
    ${SRC_ROOT}/SparseLUSolver.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseLinearSolver SofaGeneralLinearSolver SofaImplicitOdeSolver SofaSimpleFem)
target_link_libraries(${PROJECT_NAME} PUBLIC metis csparse)

sofa_create_package_with_targets(
    PACKAGE_NAME SofaSparseSolver
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS SofaSparseSolver AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "SofaSparseSolver"
    RELOCATABLE "plugins"
)
