#include "HolonomicConstraintValue.h"

#include <sofa/core/ObjectFactory.h>
#include "../utils/map.h"

namespace sofa {
namespace component {
namespace odesolver {


SOFA_DECL_CLASS(HolonomicConstraintValue);
int HolonomicConstraintValueClass = core::RegisterObject("Holonomic constraint").add< HolonomicConstraintValue >();


using namespace utils;

HolonomicConstraintValue::HolonomicConstraintValue( mstate_type* mstate )
    : Stabilization( mstate )
{}

void HolonomicConstraintValue::correction(SReal* dst, unsigned n, unsigned dim, const core::MultiVecCoordId& posId, const core::MultiVecDerivId&) const {
	assert( mstate );

    unsigned size = n*dim;

    assert( mask.getValue().empty() || mask.getValue().size() == size );
	
    mstate->copyToBuffer(dst, posId.getId(mstate.get()), size);
	
	// TODO needed ?
    map(dst, size) = -map(dst, size) / this->getContext()->getDt();

	const mask_type& mask = this->mask.getValue();
	
	// non-zero for stabilized
	unsigned i = 0;
    for(SReal* last = dst + size; dst < last; ++dst, ++i) {
		if( !mask.empty() && !mask[i] ) *dst = 0;
    }
}


void HolonomicConstraintValue::dynamics(SReal* dst, unsigned n, unsigned dim, bool /*stabilization*/, const core::MultiVecCoordId&, const core::MultiVecDerivId&) const {
	assert( mstate );

    const unsigned size = n*dim;

//    memset( dst, 0, size );

    unsigned i = 0;
    for(SReal* last = dst + size; dst < last; ++dst, ++i) {
        *dst = 0;
    }
}




}
}
}
