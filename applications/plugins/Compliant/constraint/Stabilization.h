#ifndef COMPLIANT_STABILIZATION_H
#define COMPLIANT_STABILIZATION_H

#include "BaseConstraintValue.h"

namespace sofa {
namespace component {
namespace odesolver {

// a constraint value for stabilized holonomic constraints
class SOFA_Compliant_API Stabilization : public BaseConstraintValue {
  public:

    SOFA_CLASS(Stabilization, BaseConstraintValue);

	Stabilization( mstate_type* mstate = 0);

	/// flagging which constraint lines must be stabilized (if empty, all constraints are stabilized)
	typedef vector<bool> mask_type;
	Data<mask_type> mask;
	
	// value for stabilization
	virtual void correction(SReal* dst, unsigned n) const;
	
	// value for dynamics
    virtual void dynamics(SReal* dst, unsigned n, bool stabilization) const;
	
};

}
}
}



#endif
