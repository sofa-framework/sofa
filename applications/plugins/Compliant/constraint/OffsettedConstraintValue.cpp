#include "OffsettedConstraintValue.h"

#include <sofa/core/ObjectFactory.h>
#include "../utils/map.h"

namespace sofa {
namespace component {
namespace odesolver {


SOFA_DECL_CLASS(OffsettedConstraintValue)
int OffsettedConstraintValueClass = core::RegisterObject("Constraint value abstraction").add< OffsettedConstraintValue >();


OffsettedConstraintValue::OffsettedConstraintValue( mstate_type* mstate )
    : ConstraintValue( mstate )
    , d_offset( initData(&d_offset, SReal(0.0), "offset", "Offset to add to the constraint value"))
{}


void OffsettedConstraintValue::dynamics(SReal* dst, unsigned n, unsigned dim, bool, const core::MultiVecCoordId& posId, const core::MultiVecDerivId&) const {
    assert( mstate );

    unsigned size = n*dim;

    mstate->copyToBuffer(dst, posId.getId(mstate.get()), size);

    const SReal& offset = d_offset.getValue();

    for( unsigned i=0;i<size;++i) dst[i] += offset;

	using namespace utils;
    map(dst, size) *= -1.0 / this->getContext()->getDt();
}



}
}
}
