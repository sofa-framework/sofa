#ifndef MINRESSOLVER_H
#define MINRESSOLVER_H

#include "KKTSolver.h"

#include "AssembledSystem.h"
#include <sofa/core/objectmodel/BaseObject.h>


#include "utils/minres.h"
#include "utils/cg.h"


namespace sofa {
namespace component {
namespace linearsolver {
			
// Solver for an AssembledSystem (could be non-linear in case of
// inequalities). This will eventually serve as a base class for
// all kinds of derived solver (sparse cholesky, minres, qp)
			
// TODO: base + derived classes (minres/cholesky/unilateral)
template<class NumericalSolver>
class SOFA_Compliant_API KrylovSolver : public KKTSolver {
  public:
    SOFA_ABSTRACT_CLASS(SOFA_TEMPLATE(KrylovSolver,NumericalSolver), KKTSolver);
	
    KrylovSolver();

    typedef AssembledSystem::vec vec;

	// solve the KKT system: \mat{ M - h^2 K & J^T \\ J, -C } x = rhs
	// (watch out for the compliance scaling)
	virtual void solve(vec& x,
	                   const AssembledSystem& system,
	                   const vec& rhs) const;

	virtual void factor(const AssembledSystem& system);
				
  public:
	Data<SReal> precision;
	Data<unsigned> iterations;
	Data<bool> relative;

	Data<bool> verbose;
};

class SOFA_Compliant_API MinresSolver : public KrylovSolver< ::minres<SReal> >
{
public:
  SOFA_CLASS(MinresSolver,SOFA_TEMPLATE(KrylovSolver,::minres<SReal>));
};

class SOFA_Compliant_API CgSolver : public KrylovSolver< ::cg<SReal> >
{
public:
  SOFA_CLASS(CgSolver,SOFA_TEMPLATE(KrylovSolver,::cg<SReal>));
};


}
}
}

#endif
