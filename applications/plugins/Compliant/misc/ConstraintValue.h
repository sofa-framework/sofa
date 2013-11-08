#ifndef CONSTRAINTVALUE_H
#define CONSTRAINTVALUE_H

#include <sofa/core/objectmodel/BaseObject.h>
#include "../initCompliant.h"

namespace sofa {
namespace component {
namespace odesolver {

/**

   ConstraintValue is in charge of producing right-hand side term for
   constraints (the \phi vector in the doc), depending on cases:
   elasticity, hard-stabilized constraints, restitution constraints.

   It exposes values to be mixed by the ODE solver based on
   integration scheme.

*/

class SOFA_Compliant_API ConstraintValue : public core::objectmodel::BaseObject {
	typedef core::behavior::BaseMechanicalState mstate_type;
  public:

	SOFA_CLASS(ConstraintValue, core::objectmodel::BaseObject);
	
	mstate_type::SPtr mstate;

    ConstraintValue();
	
	virtual void init();
	
	// value for stabilization
	virtual void correction(SReal* dst, unsigned n) const;
	
	// value for dynamics
	virtual void dynamics(SReal* dst, unsigned n) const;	


    Data< SReal > dampingRatio;  ///< Same damping ratio applied to all the constraints
	
};

}
}
}

#endif
