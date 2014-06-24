include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    BaseLinearSolver.h
    CGLinearSolver.h
    CGLinearSolver.inl
    CholeskySolver.h
    CholeskySolver.inl
    BTDLinearSolver.h
    BTDLinearSolver.inl
    FullVector.h
    FullMatrix.h
    DiagonalMatrix.h
    SingleMatrixAccessor.h
    SparseMatrix.h
    CompressedRowSparseMatrix.h
    CompressedRowSparseMatrix.inl
    GraphScatteredTypes.h
    DefaultMultiMatrixAccessor.h
    MatrixLinearSolver.h
    MatrixLinearSolver.inl
    MatrixExpr.h
    matrix_bloc_traits.h
    BlocMatrixWriter.h
    MinResLinearSolver.h
    MinResLinearSolver.inl

    )

set(SOURCE_FILES

    BaseLinearSolver.cpp
    CGLinearSolver.cpp
    CholeskySolver.cpp
    BTDLinearSolver.cpp
    FullVector.cpp
    GraphScatteredTypes.cpp
    DefaultMultiMatrixAccessor.cpp
    MatrixLinearSolver.cpp
    SingleMatrixAccessor.cpp
    MinResLinearSolver.cpp

    )

set(GROUP_BASE_DIR "..")

sofa_add_plugin(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_BASE_LINEAR_SOLVER")
AddLinkerDependencies(SofaSimulationTree)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
