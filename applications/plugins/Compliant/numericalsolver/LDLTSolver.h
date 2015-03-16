#ifndef COMPLIANT_LDLTSOLVER_H
#define COMPLIANT_LDLTSOLVER_H

#include "KKTSolver.h"
#include "Response.h"

#include "../utils/scoped.h"

namespace sofa {
namespace component {
namespace linearsolver {

/// Solve a dynamics system including bilateral constraints
/// with a Schur complement factorization (LDL^T Cholesky)
/// Note that the dynamics equation is solved by an external Response component
class SOFA_Compliant_API LDLTSolver : public KKTSolver {
  public:
	
	SOFA_CLASS(LDLTSolver, KKTSolver);
	
	virtual void solve(vec& x,
	                   const AssembledSystem& system,
	                   const vec& rhs) const;

	// performs factorization
	virtual void factor(const AssembledSystem& system);

    virtual void init();

	LDLTSolver();
	~LDLTSolver();


  protected:

    // response matrix
    Response::SPtr response;

    void factor_schur(const AssembledSystem& system);
    // void factor_kkt(const AssembledSystem& system) const;
    
    void solve_schur(vec& x, const AssembledSystem& system, const vec& rhs) const;
    void solve_kkt(vec& x, const AssembledSystem& system, const vec& rhs) const;
    
  private:

    struct pimpl_type;
    scoped::ptr<pimpl_type> pimpl;

    Data<bool> schur;

};


}
}
}



#endif
