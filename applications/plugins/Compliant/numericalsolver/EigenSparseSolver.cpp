#include "EigenSparseSolver.inl"

#include <sofa/core/ObjectFactory.h>



namespace sofa {
namespace component {
namespace linearsolver {




template class EigenSparseSolver< Eigen::SimplicialLDLT< AssembledSystem::cmat >, true >;
SOFA_DECL_CLASS(LDLTSolver)
static int LDLTSolverClass = core::RegisterObject("Direct LDLT solver").add< LDLTSolver >();


template class EigenSparseSolver< Eigen::SparseLU< AssembledSystem::cmat >, false >;
SOFA_DECL_CLASS(LUSolver)
static int LUSolverClass = core::RegisterObject("Direct LU solver").add< LUSolver >();


}
}
}

