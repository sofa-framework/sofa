#ifndef COMPLIANT_DAMPINGVALUE_H
#define COMPLIANT_DAMPINGVALUE_H

#include "ConstraintValue.h"

namespace sofa {
namespace component {
namespace odesolver {

/// Constraint value for damping compliance
///
/// @author Matthieu Nesme
///
class SOFA_Compliant_API DampingValue : public ConstraintValue
{
  public:

    SOFA_CLASS(DampingValue, ConstraintValue);

    DampingValue( mstate_type* mstate = NULL );

    // value for dynamics
    void dynamics(SReal* dst, unsigned n, unsigned dim, bool, const core::MultiVecCoordId& posId = core::VecCoordId::position(), const core::MultiVecDerivId& velId = core::VecDerivId::velocity()) const override;

};

}
}
}



#endif
