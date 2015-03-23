#include "EigenSparseSolver.inl"

#include <sofa/core/ObjectFactory.h>



namespace sofa {
namespace component {
namespace linearsolver {




template class SOFA_Compliant_API EigenSparseSolver< Eigen::SimplicialLDLT< AssembledSystem::cmat >, true >;
SOFA_DECL_CLASS(LDLTSolver)
static int LDLTSolverClass = core::RegisterObject("Direct LDLT solver").add< LDLTSolver >();


template class SOFA_Compliant_API EigenSparseSolver< Eigen::SparseLU< AssembledSystem::cmat >, false >;
SOFA_DECL_CLASS(LUSolver)
static int LUSolverClass = core::RegisterObject("Direct LU solver").add< LUSolver >();



/////////////////////////////////////////////


template class SOFA_Compliant_API EigenSparseIterativeSolver< Eigen::ConjugateGradient< AssembledSystem::cmat >, true >;
SOFA_DECL_CLASS(EigenCGSolver)
static int EigenCGSolverClass = core::RegisterObject("Conjugate Gradient solver").add< EigenCGSolver >();


template class SOFA_Compliant_API EigenSparseIterativeSolver< Eigen::BiCGSTAB< AssembledSystem::cmat >, false >;
SOFA_DECL_CLASS(EigenBiCGSTABSolver)
static int EigenBiCGSTABSolverClass = core::RegisterObject("Bi Conjugate Gradient stabilized solver").add< EigenBiCGSTABSolver >();


template class SOFA_Compliant_API EigenSparseIterativeSolver< Eigen::MINRES< AssembledSystem::cmat >, true >;
SOFA_DECL_CLASS(EigenMinresSolver)
static int EigenMinresSolverClass = core::RegisterObject("MINRES solver").add< EigenMinresSolver >();

}
}
}

