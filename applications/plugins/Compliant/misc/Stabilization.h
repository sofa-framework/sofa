#ifndef COMPLIANTDEV_STABILIZATION_H
#define COMPLIANTDEV_STABILIZATION_H

#include "ConstraintValue.h"

namespace sofa {
namespace component {
namespace odesolver {

// a constraint value for stabilized holonomic constraints
class SOFA_Compliant_API Stabilization : public ConstraintValue {
  public:

	SOFA_CLASS(Stabilization, ConstraintValue);

	Stabilization();

	/// flagging which constraint lines must be stabilized (if empty, all constraints are stabilized)
	typedef vector<bool> mask_type;
	Data<mask_type> mask;
	
	// value for stabilization
	virtual void correction(SReal* dst, unsigned n) const;
	
	// value for dynamics
	virtual void dynamics(SReal* dst, unsigned n) const;	
	
};

}
}
}



#endif
