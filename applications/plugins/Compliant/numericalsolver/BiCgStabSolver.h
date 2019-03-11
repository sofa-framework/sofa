#ifndef COMPLIANT_BiCgStabSolver_H
#define COMPLIANT_BiCgStabSolver_H

#include "KrylovSolver.h"

namespace sofa {
namespace component {
namespace linearsolver {

/// Bi-Conjugate Gradient Stabilized (iterative, linear solver for non-symmetric, definite matrix)
/// /// @author Matthieu Nesme
// @TODO add numerator threshold ? damping ?
class SOFA_Compliant_API BiCgStabSolver : public KrylovSolver {

  public:
    SOFA_CLASS(BiCgStabSolver, KrylovSolver);
	
    BiCgStabSolver();

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
