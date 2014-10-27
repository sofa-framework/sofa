#include "VelocityConstraintValue.h"

#include <sofa/core/ObjectFactory.h>
#include "../utils/map.h"
using std::cerr;
using std::endl;

namespace sofa {
namespace component {
namespace odesolver {


SOFA_DECL_CLASS(VelocityConstraintValue);
int VelocityConstraintValueClass = core::RegisterObject("Enforce the velocities to the given values").add< VelocityConstraintValue >();


VelocityConstraintValue::VelocityConstraintValue( mstate_type* mstate )
    : BaseConstraintValue( mstate )
    , d_velocities( initData(&d_velocities, "velocities", "The velocities to enforce"))
{
}

void VelocityConstraintValue::correction(SReal* dst, unsigned n, unsigned dim, const core::MultiVecCoordId&, const core::MultiVecDerivId&) const
{
    // no correction
    memset( dst, 0, n*dim*sizeof(SReal) );
}


void VelocityConstraintValue::dynamics(SReal* dst, unsigned n, unsigned dim, bool, const core::MultiVecCoordId& posId, const core::MultiVecDerivId&) const
{
    assert( mstate );

    unsigned size = n*dim;
    assert( d_velocities.getValue().size() == size );

    mstate->copyToBuffer(dst, posId.getId(mstate.get()), size);

	using namespace utils;
    map(dst, size) = map( &d_velocities.getValue()[0], size );
}



}
}
}
