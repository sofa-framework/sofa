#include "CgSolver.h"

#include <sofa/core/ObjectFactory.h>

#include "utils/kkt.h"
#include "utils/cg.h"

namespace sofa {
namespace component {
namespace linearsolver {

SOFA_DECL_CLASS(CgSolver);
int CgSolverClass = core::RegisterObject("Sparse CG linear solver").add< CgSolver >();

CgSolver::CgSolver() 
{
	
}

			
			
void CgSolver::factor(const AssembledSystem& sys) {
	
}



void CgSolver::solve(AssembledSystem::vec& x,
                     const AssembledSystem& system,
                     const AssembledSystem::vec& b) const {
	if( system.n ) throw std::logic_error("error: CgSolver can't handle constrained/compliant systems (yet)");

	params_type p = params(b);
	typedef cg<real> solver_type;

	kkt::matrixQ A(system);

	solver_type::solve(x, A, b, p);
}
			
}
}
}


