#include "BiCgStabSolver.h"

#include <sofa/core/ObjectFactory.h>


#include "../utils/schur.h"
#include "../utils/kkt.h"
#include "../utils/bicgstab.h"


namespace sofa {
namespace component {
namespace linearsolver {

SOFA_DECL_CLASS(BiCgStabSolver)
const int BiCgStabSolverClass = core::RegisterObject("Sparse biCGStab linear solver").add< BiCgStabSolver >();

BiCgStabSolver::BiCgStabSolver()
{
	
}


void BiCgStabSolver::solve_schur_impl(vec& lambda,
                                    const schur_type& A,
                                    const vec& b,
                                    params_type& p) const{
    typedef bicgstab<SReal> solver_type;		
    solver_type::solve(lambda, A, b, p);
}


void BiCgStabSolver::solve_kkt_impl(vec& x,
                                  const kkt_type& A,
                                  const vec& b,
                                  params_type& p) const{
    typedef bicgstab<real> solver_type;
	solver_type::solve(x, A, b, p);
}



const char* BiCgStabSolver::method() const { return "bicgstab"; }



}
}
}


