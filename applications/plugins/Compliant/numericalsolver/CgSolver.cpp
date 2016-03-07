#include "CgSolver.h"

#include <sofa/core/ObjectFactory.h>


#include "../utils/schur.h"
#include "../utils/kkt.h"
#include "../utils/cg.h"
// #include "utils/preconditionedcg.h"


namespace sofa {
namespace component {
namespace linearsolver {

SOFA_DECL_CLASS(CgSolver)
const int CgSolverClass = core::RegisterObject("Sparse CG linear solver").add< CgSolver >();

CgSolver::CgSolver() 
{
	
}


void CgSolver::solve_schur_impl(vec& lambda,
                                    const schur_type& A,
                                    const vec& b,
                                    params_type& p) const{
    typedef ::cg<SReal> solver_type;		
    solver_type::solve(lambda, A, b, p);
}


void CgSolver::solve_kkt_impl(vec& x,
                                  const kkt_type& A,
                                  const vec& b,
                                  params_type& p) const{
    typedef ::cg<real> solver_type;
	solver_type::solve(x, A, b, p);
}



const char* CgSolver::method() const { return "cg"; }


}
}
}


