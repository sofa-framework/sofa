#include "EigenSparseSolver.inl"

#include <sofa/core/ObjectFactory.h>



namespace sofa {
namespace component {
namespace linearsolver {




template class SOFA_Compliant_API EigenSparseSolver< LDLTSparseLinearSolver, true >;
SOFA_DECL_CLASS(LDLTSolver)
int LDLTSolverClass = core::RegisterObject("Direct LDLT solver").add< LDLTSolver >();


template class SOFA_Compliant_API EigenSparseSolver< LUSparseLinearSolver >;
SOFA_DECL_CLASS(LUSolver)
int LUSolverClass = core::RegisterObject("Direct LU solver").add< LUSolver >();



/////////////////////////////////////////////


template class SOFA_Compliant_API EigenSparseIterativeSolver< CGSparseLinearSolver, true >;
SOFA_DECL_CLASS(EigenCGSolver)
int EigenCGSolverClass = core::RegisterObject("Conjugate Gradient solver").add< EigenCGSolver >();


template class SOFA_Compliant_API EigenSparseIterativeSolver< BiCGSTABSparseLinearSolver >;
SOFA_DECL_CLASS(EigenBiCGSTABSolver)
int EigenBiCGSTABSolverClass = core::RegisterObject("Bi Conjugate Gradient stabilized solver").add< EigenBiCGSTABSolver >();


template class SOFA_Compliant_API EigenSparseIterativeSolver< MINRESSparseLinearSolver, true >;
SOFA_DECL_CLASS(EigenMINRESSolver)
int EigenMinresSolverClass = core::RegisterObject("MINRES solver").add< EigenMINRESSolver >();


template class SOFA_Compliant_API EigenSparseIterativeSolver< GMRESSparseLinearSolver >;
SOFA_DECL_CLASS(EigenGMRESSolver)
int EigenGmresSolverClass = core::RegisterObject("GMRES solver").add< EigenGMRESSolver >();


}
}
}

