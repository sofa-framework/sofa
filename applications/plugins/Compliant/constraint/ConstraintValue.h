#ifndef CONSTRAINTVALUE_H
#define CONSTRAINTVALUE_H

#include "BaseConstraintValue.h"

namespace sofa {
namespace component {
namespace odesolver {

/**

   ConstraintValue for elasticity (compliant constraints)

*/

class SOFA_Compliant_API ConstraintValue : public BaseConstraintValue {
  public:

    SOFA_CLASS(ConstraintValue, BaseConstraintValue);

    ConstraintValue( mstate_type* mstate = 0 );

	// value for stabilization
    void correction(SReal* dst, unsigned n, unsigned dim, const core::MultiVecCoordId& posId = core::VecCoordId::position(), const core::MultiVecDerivId& velId = core::VecDerivId::velocity()) const override;
	
	// value for dynamics
    void dynamics(SReal* dst, unsigned n, unsigned dim, bool, const core::MultiVecCoordId& posId = core::VecCoordId::position(), const core::MultiVecDerivId& velId = core::VecDerivId::velocity()) const override;

};

}
}
}

#endif
