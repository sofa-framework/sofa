cmake_minimum_required(VERSION 2.8.12)
project(SofaBaseLinearSolver)

set(HEADER_FILES
    BTDLinearSolver.h
    BTDLinearSolver.inl
    BlocMatrixWriter.h
    CGLinearSolver.h
    CGLinearSolver.inl
    CholeskySolver.h
    CholeskySolver.inl
    CompressedRowSparseMatrix.h
    CompressedRowSparseMatrix.inl
    DefaultMultiMatrixAccessor.h
    DiagonalMatrix.h
    FullMatrix.h
    FullVector.h
    GraphScatteredTypes.h
    MatrixExpr.h
    MatrixLinearSolver.h
    MatrixLinearSolver.inl
    MinResLinearSolver.h
    MinResLinearSolver.inl
    SingleMatrixAccessor.h
    SparseMatrix.h
    config.h
    initBaseLinearSolver.h
    matrix_bloc_traits.h
)

set(SOURCE_FILES
    BTDLinearSolver.cpp
    CGLinearSolver.cpp
    CholeskySolver.cpp
    DefaultMultiMatrixAccessor.cpp
    FullVector.cpp
    GraphScatteredTypes.cpp
    MatrixLinearSolver.cpp
    MinResLinearSolver.cpp
    SingleMatrixAccessor.cpp
    initBaseLinearSolver.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationCommon)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_BASE_LINEAR_SOLVER")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

SofaBase_install(${PROJECT_NAME})
