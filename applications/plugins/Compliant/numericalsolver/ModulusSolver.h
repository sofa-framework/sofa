#ifndef COMPLIANT_MODULUSSOLVER_H
#define COMPLIANT_MODULUSSOLVER_H

#include "IterativeSolver.h"
#include "Response.h"
#include "SubKKT.h"

namespace sofa {
namespace component {
namespace linearsolver {

// solve unilateral sparse systems without forming the schur
// complement
class SOFA_Compliant_API ModulusSolver : public IterativeSolver {
  public:
	
	SOFA_CLASS(ModulusSolver, IterativeSolver);
	
	virtual void solve(vec& x,
	                   const AssembledSystem& system,
	                   const vec& rhs) const;

	virtual void factor(const AssembledSystem& system);

    virtual void init();

	ModulusSolver();
	~ModulusSolver();
    
  protected:
    
    // response matrix
    Response::SPtr response;

    SubKKT sub;
    Data<real> omega;

    Data<unsigned> anderson;
    
  private:
    vec unilateral, diagonal;
    
};


}
}
}



#endif
