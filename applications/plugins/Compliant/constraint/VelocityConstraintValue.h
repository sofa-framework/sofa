#ifndef VELOCITYCONSTRAINTVALUE_H
#define VELOCITYCONSTRAINTVALUE_H

#include "BaseConstraintValue.h"

namespace sofa {
namespace component {
namespace odesolver {

/**

   Enforce the velocities to the given values

   @author Matthieu Nesme

*/

class SOFA_Compliant_API VelocityConstraintValue : public BaseConstraintValue {
  public:

    SOFA_CLASS(VelocityConstraintValue, BaseConstraintValue);

    VelocityConstraintValue( mstate_type* mstate = 0 );

	// value for stabilization
    void correction(SReal* dst, unsigned n, unsigned dim, const core::MultiVecCoordId& posId = core::VecCoordId::position(), const core::MultiVecDerivId& velId = core::VecDerivId::velocity()) const override;
	
	// value for dynamics
    void dynamics(SReal* dst, unsigned n, unsigned dim, bool, const core::MultiVecCoordId& posId = core::VecCoordId::position(), const core::MultiVecDerivId& velId = core::VecDerivId::velocity()) const override;


    Data< helper::vector<SReal> > d_velocities;  ///< fixed velocities
	
};

}
}
}

#endif
