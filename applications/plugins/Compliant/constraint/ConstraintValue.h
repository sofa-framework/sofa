#ifndef CONSTRAINTVALUE_H
#define CONSTRAINTVALUE_H

#include "BaseConstraintValue.h"
#include "../initCompliant.h"

namespace sofa {
namespace component {
namespace odesolver {

/**

   Default ConstraintValue for damped elasticity

*/

class SOFA_Compliant_API ConstraintValue : public BaseConstraintValue {
  public:

    SOFA_CLASS(ConstraintValue, BaseConstraintValue);

    ConstraintValue() {}
    ConstraintValue( mstate_type* mstate );

	// value for stabilization
	virtual void correction(SReal* dst, unsigned n) const;
	
	// value for dynamics
	virtual void dynamics(SReal* dst, unsigned n) const;	


//    Data< SReal > dampingRatio;  ///< Same damping ratio applied to all the constraints
	
};

}
}
}

#endif
