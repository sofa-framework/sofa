#ifndef COMPLIANT_CGSOLVER_H
#define COMPLIANT_CGSOLVER_H

#include "KrylovSolver.h"

namespace sofa {
namespace component {
namespace linearsolver {

/// Conjugate Gradient (iterative, linear solver for symmetric, definite matrix)
// TODO add numerator threshold ? damping ?
class SOFA_Compliant_API CgSolver : public KrylovSolver {

  public:
	SOFA_CLASS(CgSolver, KrylovSolver);
	
	CgSolver();				

  protected:

    virtual void solve_schur_impl(vec& lambda,
                                  const schur_type& A,
                                  const vec& b,
                                  params_type& p) const;

    virtual void solve_kkt_impl(vec& x,
                                const kkt_type& A,
                                const vec& b,
                                params_type& p) const;

    virtual const char* method() const;
    
};


}
}
}

#endif
