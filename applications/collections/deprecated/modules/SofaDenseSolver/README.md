# SofaDenseSolver

Plugin with Solvers using dense matrices.
Needs the library NewMat.

Using the template "NewMat" instead of the default ones, it allows the user to use NewMat's matrices for:

- CGLinearSolver (specialization from SofaBaseLinearSolver),
- CholeskySolver (specialization from SofaBaseLinearSolver).

e.g in a XML scene file: 
```
<CGLinearSolver template="NewMat" iterations="25" />
```
Moreover, it introduces the solver LULinearSolver (only on NewMat matrices).