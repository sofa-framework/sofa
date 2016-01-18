#include "BaumgarteStabilization.h"

#include <sofa/core/ObjectFactory.h>
#include "../utils/map.h"

namespace sofa {
namespace component {
namespace odesolver {


SOFA_DECL_CLASS(BaumgarteStabilization)
int BaumgarteStabilizationClass = core::RegisterObject("Constraint value for Baumgarte stabilization").add< BaumgarteStabilization >();


BaumgarteStabilization::BaumgarteStabilization( mstate_type* mstate )
    : ConstraintValue( mstate )
    , d_alpha( initData(&d_alpha, SReal(0), "alpha", "The constraint violation coefficient"))
{
}


void BaumgarteStabilization::dynamics(SReal* dst, unsigned n, unsigned dim, bool, const core::MultiVecCoordId& posId, const core::MultiVecDerivId&) const {
    assert( mstate );

    unsigned size = n*dim;

    mstate->copyToBuffer(dst, posId.getId(mstate.get()), size);

	using namespace utils;
    map(dst, size) = -map(dst, size) * d_alpha.getValue();
}



}
}
}
