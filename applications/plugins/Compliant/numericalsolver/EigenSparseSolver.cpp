#include "EigenSparseSolver.inl"

#include <sofa/core/ObjectFactory.h>



namespace sofa {
namespace component {
namespace linearsolver {




template class SOFA_Compliant_API EigenSparseSolver< LDLTSparseLinearSolver, true >;
int LDLTSolverClass = core::RegisterObject("Direct LDLT solver").add< LDLTSolver >();

template class SOFA_Compliant_API EigenSparseSolver< LLTSparseLinearSolver, true >;
int LLTSolverClass = core::RegisterObject("Direct LLT solver").add< LLTSolver >();

template class SOFA_Compliant_API EigenSparseSolver< LUSparseLinearSolver >;
int LUSolverClass = core::RegisterObject("Direct LU solver").add< LUSolver >();



/////////////////////////////////////////////


template class SOFA_Compliant_API EigenSparseIterativeSolver< CGSparseLinearSolver, true >;
int EigenCGSolverClass = core::RegisterObject("Conjugate Gradient solver").add< EigenCGSolver >();


template class SOFA_Compliant_API EigenSparseIterativeSolver< BiCGSTABSparseLinearSolver >;
int EigenBiCGSTABSolverClass = core::RegisterObject("Bi Conjugate Gradient stabilized solver").add< EigenBiCGSTABSolver >();


template class SOFA_Compliant_API EigenSparseIterativeSolver< MINRESSparseLinearSolver, true >;
int EigenMinresSolverClass = core::RegisterObject("MINRES solver").add< EigenMINRESSolver >();


template class SOFA_Compliant_API EigenSparseIterativeSolver< GMRESSparseLinearSolver >;
int EigenGmresSolverClass = core::RegisterObject("GMRES solver").add< EigenGMRESSolver >();


}
}
}

