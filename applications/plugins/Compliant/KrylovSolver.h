#ifndef COMPLIANT_KRYLOVSOLVER_H
#define COMPLIANT_KRYLOVSOLVER_H

#include "KKTSolver.h"
#include "Response.h"

#include "utils/krylov.h"

#include "AssembledSystem.h"
#include <sofa/core/objectmodel/BaseObject.h>

#include <Eigen/SparseCholesky>

namespace sofa {
namespace component {
namespace linearsolver {


// should be IterativeSolver -> KrylovSolver -> ...
class SOFA_Compliant_API KrylovSolver : public KKTSolver {
  public:
	
	KrylovSolver();				
	
	Data<SReal> precision;
	Data<unsigned> iterations;
	Data<bool> relative;
	
	Data<bool> schur;
	
	Data<bool> verbose;
	
	virtual void init();
	
	virtual void solve(vec& x,
	                   const system_type& system,
	                   const vec& rhs) const;

	virtual void factor(const system_type& sys);
	
  protected:

	virtual void solve_schur(vec& x,
	                         const system_type& system,
	                         const vec& rhs) const = 0;
	
	virtual void solve_kkt(vec& x,
	                       const system_type& system,
	                       const vec& rhs) const = 0;
	
	typedef ::krylov<SReal>::params params_type;
	
	virtual params_type params(const vec& rhs) const;

	typedef Response response_type;
	Response::SPtr response;
};


}
}
}

#endif
