#ifndef LDLTSOLVER_H
#define LDLTSOLVER_H

#include "KKTSolver.h"
#include "Response.h"

#include <Eigen/SparseCholesky>


#include "utils/thread_variable.h"
#include "SubKKT.h"

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

  private:
	
	struct pimpl_type {
        typedef Eigen::SimplicialLDLT< cmat >  solver_type;

        solver_type solver;
        cmat HinvPJT;
        cmat schur;
	};

    void factor_schur( const cmat& schur );

	
    mutable thread_variable<pimpl_type> pimpl;

    SubKKT sub;
};


}
}
}



#endif
