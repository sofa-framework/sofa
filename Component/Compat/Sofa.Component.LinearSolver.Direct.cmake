set(SOFAGENERALLINEARSOLVER_SRC src/SofaGeneralLinearSolver)
set(SOFASPARSESOLVER_SRC src/SofaSparseSolver)
set(SOFADENSESOLVER_SRC src/SofaDenseSolver)

list(APPEND HEADER_FILES
    ${SOFAGENERALLINEARSOLVER_SRC}/BTDLinearSolver.h
    ${SOFAGENERALLINEARSOLVER_SRC}/BTDLinearSolver.inl
    ${SOFAGENERALLINEARSOLVER_SRC}/CholeskySolver.h
    ${SOFAGENERALLINEARSOLVER_SRC}/CholeskySolver.inl
    ${SOFADENSESOLVER_SRC}/SVDLinearSolver.h
    ${SOFASPARSESOLVER_SRC}/PrecomputedLinearSolver.h
    ${SOFASPARSESOLVER_SRC}/PrecomputedLinearSolver.inl
    ${SOFASPARSESOLVER_SRC}/SparseLDLSolver.h
    ${SOFASPARSESOLVER_SRC}/SparseLDLSolver.inl
    ${SOFASPARSESOLVER_SRC}/SparseLDLSolverImpl.h
    ${SOFASPARSESOLVER_SRC}/SparseCholeskySolver.h
    ${SOFASPARSESOLVER_SRC}/SparseLUSolver.h
    ${SOFASPARSESOLVER_SRC}/SparseLUSolver.inl
    ${SOFASPARSESOLVER_SRC}/FillReducingOrdering.h
    ${SOFASPARSESOLVER_SRC}/FillReducingOrdering.inl
)
