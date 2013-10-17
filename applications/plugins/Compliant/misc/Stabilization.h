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

	/// flagging which constraint lines must be stabilized (if empty, all constraints are stabilized)
	std::vector<bool> mask;
	
	// value for stabilization
	virtual void correction(SReal* dst, unsigned n) const;
	
	// value for dynamics
	virtual void dynamics(SReal* dst, unsigned n) const;	
	
};

}
}
}



#endif
