#include "DampingValue.h"

#include <sofa/core/ObjectFactory.h>
#include "../utils/map.h"

namespace sofa {
namespace component {
namespace odesolver {


SOFA_DECL_CLASS(DampingValue)
int DampingValueClass = core::RegisterObject("Constraint value for damping compliance").add< DampingValue >();


DampingValue::DampingValue( mstate_type* mstate )
    : ConstraintValue( mstate )
{}

using namespace utils;

void DampingValue::dynamics(SReal* dst, unsigned n, unsigned dim, bool, const core::MultiVecCoordId&, const core::MultiVecDerivId& velId) const
{
    assert( mstate );

    unsigned size = n*dim;

    // we sneakily fake constraint error with reflected-adjusted relative velocity
    mstate->copyToBuffer(dst, velId.getId(mstate.get()), size);
    map(dst, size) *= -1;
}






}
}
}
