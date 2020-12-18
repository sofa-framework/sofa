cmake_minimum_required(VERSION 3.12)
project(SofaBaseLinearSolver LANGUAGES CXX)

set(SOFABASELINEARSOLVER_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFABASELINEARSOLVER_SRC}/config.h.in
    ${SOFABASELINEARSOLVER_SRC}/initSofaBaseLinearSolver.h
    ${SOFABASELINEARSOLVER_SRC}/BlocMatrixWriter.h
    ${SOFABASELINEARSOLVER_SRC}/CGLinearSolver.h
    ${SOFABASELINEARSOLVER_SRC}/CGLinearSolver.inl
    ${SOFABASELINEARSOLVER_SRC}/CompressedRowSparseMatrix.h
    ${SOFABASELINEARSOLVER_SRC}/CompressedRowSparseMatrix.inl
    ${SOFABASELINEARSOLVER_SRC}/DefaultMultiMatrixAccessor.h
    ${SOFABASELINEARSOLVER_SRC}/DiagonalMatrix.h
    ${SOFABASELINEARSOLVER_SRC}/FullMatrix.h
    ${SOFABASELINEARSOLVER_SRC}/FullVector.h
    ${SOFABASELINEARSOLVER_SRC}/GraphScatteredTypes.h
    ${SOFABASELINEARSOLVER_SRC}/MatrixExpr.h
    ${SOFABASELINEARSOLVER_SRC}/MatrixLinearSolver.h
    ${SOFABASELINEARSOLVER_SRC}/MatrixLinearSolver.inl
    ${SOFABASELINEARSOLVER_SRC}/SingleMatrixAccessor.h
    ${SOFABASELINEARSOLVER_SRC}/SparseMatrix.h
    ${SOFABASELINEARSOLVER_SRC}/matrix_bloc_traits.h
)

set(SOURCE_FILES
    ${SOFABASELINEARSOLVER_SRC}/initSofaBaseLinearSolver.cpp
    ${SOFABASELINEARSOLVER_SRC}/CGLinearSolver.cpp
    ${SOFABASELINEARSOLVER_SRC}/DefaultMultiMatrixAccessor.cpp
    ${SOFABASELINEARSOLVER_SRC}/FullVector.cpp
    ${SOFABASELINEARSOLVER_SRC}/GraphScatteredTypes.cpp
    ${SOFABASELINEARSOLVER_SRC}/MatrixLinearSolver.cpp
    ${SOFABASELINEARSOLVER_SRC}/SingleMatrixAccessor.cpp
)

find_package(SofaFramework REQUIRED) # SofaSimulationCore

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationCore)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFABASELINEARSOLVER_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFABASELINEARSOLVER_BUILD_TESTS)
    enable_testing()
    add_subdirectory(${PROJECT_NAME}_test)
endif()
