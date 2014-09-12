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


void HolonomicConstraintValue::dynamics(SReal* dst, unsigned n, unsigned dim, bool /*stabilization*/, const core::MultiVecCoordId&, const core::MultiVecDerivId&) const {
	assert( mstate );

    const unsigned size = n*dim;

    memset( dst, 0, size*sizeof(SReal) );
}




}
}
}
