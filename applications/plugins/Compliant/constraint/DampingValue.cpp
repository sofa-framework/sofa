#include "DampingValue.h"

#include <sofa/core/ObjectFactory.h>
#include "../utils/map.h"

namespace sofa {
namespace component {
namespace odesolver {


SOFA_DECL_CLASS(DampingValue);
int DampingValueClass = core::RegisterObject("Constraint value for damping compliance").add< DampingValue >();


DampingValue::DampingValue( mstate_type* mstate )
    : ConstraintValue( mstate )
{}


void DampingValue::dynamics(SReal* dst, unsigned n) const
{
    assert( mstate );

    // we sneakily fake constraint error with reflected-adjusted relative velocity
    mstate->copyToBuffer(dst, core::VecDerivId::velocity(), n);
    mapToEigen(dst, n) = -mapToEigen(dst, n);
}






}
}
}
