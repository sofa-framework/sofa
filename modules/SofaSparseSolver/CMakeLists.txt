cmake_minimum_required(VERSION 3.1)
project(SofaSparseSolver VERSION 1.0 LANGUAGES CXX)

# Dependencies
find_package(SofaBase REQUIRED) # SofaGeneralLinearSolver
find_package(SofaCommon REQUIRED) # SofaImplicitOdeSolver SofaSimpleFem
find_package(SofaGeneral REQUIRED) # SofaGeneralLinearSolver

add_subdirectory(extlibs/csparse)
find_package(metis QUIET) # Unix users can have an installed version of metis
if(NOT metis_FOUND)
    message(STATUS "SofaSparseSolver: metis library not found, will use built-in metis library.")
    add_subdirectory(extlibs/metis-5.1.0)
endif()

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

sofa_generate_package(
    NAME SofaSparseSolver
    VERSION ${PROJECT_VERSION}
    TARGETS SofaSparseSolver
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "SofaSparseSolver"
    RELOCATABLE "plugins"
)
