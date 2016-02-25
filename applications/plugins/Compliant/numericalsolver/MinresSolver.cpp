#include "MinresSolver.h"

#include <sofa/core/ObjectFactory.h>

#include "../utils/scoped.h"
#include "../utils/minres.h"

#include "../utils/kkt.h"
#include "../utils/schur.h"

namespace sofa {
namespace component {
namespace linearsolver {
 
SOFA_DECL_CLASS(MinresSolver)
const int MinresSolverClass = core::RegisterObject("Sparse Minres linear solver").add< MinresSolver >();



void MinresSolver::solve_schur_impl(vec& lambda,
                                    const schur_type& A,
                                    const vec& b,
                                    params_type& p) const{
    typedef ::minres<SReal> solver_type;		
    solver_type::solve(lambda, A, b, p);
}


void MinresSolver::solve_kkt_impl(vec& x,
                                  const kkt_type& A,
                                  const vec& b,
                                  params_type& p) const {
    typedef minres<real> solver_type;
	solver_type::solve(x, A, b, p);
}


const char* MinresSolver::method() const { return "minres"; }
			
}
}
}


