#ifndef COMPLIANT_NUMSOLVER_KKTSOLVER_H
#define COMPLIANT_NUMSOLVER_KKTSOLVER_H

#include "initCompliant.h"

#include "assembly/AssembledSystem.h"
#include "preconditioner/BasePreconditioner.h"
#include <sofa/core/behavior/LinearSolver.h>

namespace sofa {
namespace component {
namespace linearsolver {
			
// Solver for an AssembledSystem (could be non-linear in case of
// inequalities). This will eventually serve as a base class for
// all kinds of derived solver (sparse cholesky, minres, qp)



//#define GR_BENCHMARK

			
// TODO: base + derived classes (minres/cholesky/unilateral)
class SOFA_Compliant_API KKTSolver : public core::behavior::BaseLinearSolver {
  public:
	SOFA_CLASS(KKTSolver, core::objectmodel::BaseObject);

	// solve the KKT system: \mat{ M - h^2 K & J^T \\ J, -C } x = rhs
	// (watch out for the compliance scaling)
	
	typedef AssembledSystem system_type;
	
	typedef system_type::real real;
    typedef system_type::vec vec;
    typedef system_type::mat mat;

    Data<bool> debug; ///< print debug info

    KKTSolver()
       : debug(initData(&debug,false,"debug","print debug info"))
       , _preconditioner( NULL )
    {}

    virtual void init()
    {
        // look for an optional preconditioner
        _preconditioner = this->getContext()->get<preconditioner_type>(core::objectmodel::BaseContext::Local);
    }
	
	virtual void factor(const system_type& system) = 0;
	
	virtual void solve(vec& x,
	                   const system_type& system,
                       const vec& rhs) const = 0;


    virtual void solveWithPreconditioner(vec& x,
                       const system_type& system,
                       const vec& rhs) const
    {
        if( _preconditioner ) serr<<"The preconditioner won't be used by this numerical solver\n";
        solve( x, system, rhs );
    }

    // return true if the solver can only handle equality constraints (in opposition with LCP for instance)
    virtual bool isLinear() const { return true; }

    /// By default, it does nothing, but for some LCP solvers, it is useful to distinguish between dynamics vs correction passes (not to perform the same constraint projection for instance)
    virtual void setCorrectionPass( bool ){}

    
#ifdef GR_BENCHMARK
    mutable unsigned nbiterations;
#endif

protected:

    typedef linearsolver::BasePreconditioner preconditioner_type;
    preconditioner_type* _preconditioner;

};


}
}
}

#endif
