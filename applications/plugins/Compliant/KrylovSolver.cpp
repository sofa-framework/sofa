#include "KrylovSolver.inl"

#include <sofa/core/ObjectFactory.h>


namespace sofa {
namespace component {
namespace linearsolver {


SOFA_DECL_CLASS(MinresSolver);
int MinresSolverClass = core::RegisterObject("Sparse Minres linear solver").add< MinresSolver >();

SOFA_DECL_CLASS(CgSolver);
int CgSolverClass = core::RegisterObject("Sparse Conjugate Gradient linear solver").add< CgSolver >();

			
}
}
}


