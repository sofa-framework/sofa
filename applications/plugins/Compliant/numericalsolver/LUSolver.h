#ifndef COMPLIANT_LUSOLVER_H
#define COMPLIANT_LUSOLVER_H

#include "KKTSolver.h"
#include "Response.h"


#include <Eigen/SparseLU>

#include "../utils/thread_variable.h"


namespace sofa {
namespace component {
namespace linearsolver {

/// Solve a dynamics system including bilateral constraints
/// with a Schur complement factorization (LU)
/// Note that the dynamics equation is solved by an external Response component
class SOFA_Compliant_API LUSolver : public KKTSolver {
  public:
	
    SOFA_CLASS(LUSolver, KKTSolver);
	
	virtual void solve(vec& x,
	                   const AssembledSystem& system,
	                   const vec& rhs) const;

	// performs factorization
	virtual void factor(const AssembledSystem& system);

    virtual void init();

    LUSolver();
    ~LUSolver();


  protected:

    // response matrix
    Response::SPtr response;

  private:
	
	struct pimpl_type {
		typedef SReal real;
		typedef Eigen::SparseMatrix<real, Eigen::ColMajor> cmat;
		typedef Eigen::SparseMatrix<real, Eigen::RowMajor> rmat;
		
        typedef Eigen::SparseLU< cmat > solver_type;

        solver_type schur;
        cmat HinvPJT;
		
	};

    void factor_schur( const pimpl_type::cmat& schur );

	
    mutable thread_variable<pimpl_type> pimpl;

};


}
}
}



#endif
