#ifndef COMPLIANT_KRYLOVSOLVER_H
#define COMPLIANT_KRYLOVSOLVER_H

#include "KKTSolver.h"
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
	
	Data<bool> verbose;
	
  protected:

	typedef ::krylov<SReal>::params params_type;
	
	virtual params_type params(const vec& rhs) const;
};


}
}
}

#endif
