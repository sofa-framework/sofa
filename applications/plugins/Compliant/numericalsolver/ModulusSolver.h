#ifndef COMPLIANT_MODULUSSOLVER_H
#define COMPLIANT_MODULUSSOLVER_H

#include "IterativeSolver.h"
#include "Response.h"
#include "SubKKT.h"

namespace sofa {
namespace component {
namespace linearsolver {

/// Solve a dynamics system including bilateral constraints
/// with a Schur complement factorization (LDL^T Cholesky)
/// Note that the dynamics equation is solved by an external Response component
class SOFA_Compliant_API ModulusSolver : public IterativeSolver {
  public:
	
	SOFA_CLASS(ModulusSolver, KKTSolver);
	
	virtual void solve(vec& x,
	                   const AssembledSystem& system,
	                   const vec& rhs) const;

	// performs factorization
	virtual void factor(const AssembledSystem& system);

    virtual void init();

	ModulusSolver();
	~ModulusSolver();
    
  protected:

    // response matrix
    Response::SPtr response;

    SubKKT sub;
    Data<real> omega;
    
  private:
    vec unilateral, diagonal;
    
};


}
}
}



#endif
