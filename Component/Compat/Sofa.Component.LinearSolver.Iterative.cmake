set(SOFABASELINEARSOLVER_SRC src/SofaBaseLinearSolver)
set(SOFAGENERALLINEARSOLVER_SRC src/SofaGeneralLinearSolver)
set(SOFAPRECONDITIONER_SRC src/SofaPreconditioner)

list(APPEND HEADER_FILES
    ${SOFABASELINEARSOLVER_SRC}/CGLinearSolver.h
    ${SOFABASELINEARSOLVER_SRC}/CGLinearSolver.inl
    ${SOFAGENERALLINEARSOLVER_SRC}/MinResLinearSolver.h
    ${SOFAGENERALLINEARSOLVER_SRC}/MinResLinearSolver.inl
    ${SOFAPRECONDITIONER_SRC}/ShewchukPCGLinearSolver.h
    
    ${SOFABASELINEARSOLVER_SRC}/BlocMatrixWriter.h
    ${SOFABASELINEARSOLVER_SRC}/DefaultMultiMatrixAccessor.h
    ${SOFABASELINEARSOLVER_SRC}/GraphScatteredTypes.h
    ${SOFABASELINEARSOLVER_SRC}/MatrixLinearSolver.h
    ${SOFABASELINEARSOLVER_SRC}/MatrixLinearSolver.inl
    ${SOFABASELINEARSOLVER_SRC}/SingleMatrixAccessor.h
)
