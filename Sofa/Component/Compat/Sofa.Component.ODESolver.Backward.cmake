set(SOFAIMPLICITODESOLVER_SRC src/SofaImplicitOdeSolver)
set(SOFAGENERALIMPLICITODESOLVER_SRC src/SofaGeneralImplicitOdeSolver)
set(SOFAMISCSOLVER_SRC src/SofaMiscSolver)

list(APPEND HEADER_FILES
    ${SOFAIMPLICITODESOLVER_SRC}/EulerImplicitSolver.h
    ${SOFAIMPLICITODESOLVER_SRC}/StaticSolver.h
    ${SOFAGENERALIMPLICITODESOLVER_SRC}/VariationalSymplecticSolver.h
    ${SOFAMISCSOLVER_SRC}/NewmarkImplicitSolver.h
)
