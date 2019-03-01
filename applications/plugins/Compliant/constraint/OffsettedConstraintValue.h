#ifndef OFFSETTEDCONSTRAINTVALUE_H
#define OFFSETTEDCONSTRAINTVALUE_H

#include "ConstraintValue.h"

namespace sofa {
namespace component {
namespace odesolver {

/**

   The same as ConstraintValue (default value for damped elasticity) with an offset (added)

   @author Matthieu Nesme
*/

class SOFA_Compliant_API OffsettedConstraintValue : public ConstraintValue {
  public:

    SOFA_CLASS(OffsettedConstraintValue, ConstraintValue);

    OffsettedConstraintValue( mstate_type* mstate = 0 );

	// value for dynamics
    void dynamics(SReal* dst, unsigned n, unsigned dim, bool, const core::MultiVecCoordId& posId = core::VecCoordId::position(), const core::MultiVecDerivId& velId = core::VecDerivId::velocity()) const override;

    /// the offset value to add
    Data<SReal> d_offset;
};

}
}
}

#endif
