#ifndef __BaumgarteStabilization_H
#define __BaumgarteStabilization_H

#include "ConstraintValue.h"

namespace sofa {
namespace component {
namespace odesolver {

/**

   ConstraintValue for Baumgarte stabilization

   Assuming a velocity formulation and a holonomic constraint: \dot g = Jv = 0
   Baumgarte stabilization adds a term depending on current violation g: \dot g = Jv + \alpha g = 0

    @author Matthieu Nesme
*/

class SOFA_Compliant_API BaumgarteStabilization : public ConstraintValue {
  public:

    SOFA_CLASS(BaumgarteStabilization, ConstraintValue);

    BaumgarteStabilization( mstate_type* mstate = 0 );

	// value for dynamics
    virtual void dynamics(SReal* dst, unsigned n, unsigned dim, bool, const core::MultiVecCoordId& posId = core::VecCoordId::position(), const core::MultiVecDerivId& velId = core::VecDerivId::velocity()) const;

    Data< SReal > d_alpha; ///< The constraint violation coefficient
};

}
}
}

#endif
