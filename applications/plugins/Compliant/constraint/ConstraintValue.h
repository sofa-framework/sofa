#ifndef CONSTRAINTVALUE_H
#define CONSTRAINTVALUE_H

#include "BaseConstraintValue.h"

namespace sofa {
namespace component {
namespace odesolver {

/**

   Default ConstraintValue for damped elasticity

*/

class SOFA_Compliant_API ConstraintValue : public BaseConstraintValue {
  public:

    SOFA_CLASS(ConstraintValue, BaseConstraintValue);

    ConstraintValue( mstate_type* mstate = 0 );

	// value for stabilization
    virtual void correction(SReal* dst, unsigned n, unsigned dim, const core::MultiVecCoordId& posId = core::VecCoordId::position(), const core::MultiVecDerivId& velId = core::VecDerivId::velocity()) const;
	
	// value for dynamics
    virtual void dynamics(SReal* dst, unsigned n, unsigned dim, bool, const core::MultiVecCoordId& posId = core::VecCoordId::position(), const core::MultiVecDerivId& velId = core::VecDerivId::velocity()) const;


//    Data< SReal > dampingRatio;  ///< Same damping ratio applied to all the constraints
	
};

}
}
}

#endif
