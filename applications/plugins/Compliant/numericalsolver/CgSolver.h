#ifndef COMPLIANT_CGSOLVER_H
#define COMPLIANT_CGSOLVER_H

#include "KrylovSolver.h"

namespace sofa {
namespace component {
namespace linearsolver {

/// Conjugate Gradient (iterative, linear solver for symmetric, definite matrix)
/// @author Matthieu Nesme
// TODO add numerator threshold ? damping ?
class SOFA_Compliant_API CgSolver : public KrylovSolver {

  public:
	SOFA_CLASS(CgSolver, KrylovSolver);
	
	CgSolver();				

  protected:

    void solve_schur_impl(vec& lambda,
                                  const schur_type& A,
                                  const vec& b,
                                  params_type& p) const override;

    void solve_kkt_impl(vec& x,
                                const kkt_type& A,
                                const vec& b,
                                params_type& p) const override;

    const char* method() const override;
    
};


}
}
}

#endif
