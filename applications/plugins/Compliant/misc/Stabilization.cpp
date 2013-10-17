#include "Stabilization.h"

#include <sofa/core/ObjectFactory.h>

namespace sofa {
namespace component {
namespace odesolver {


SOFA_DECL_CLASS(Stabilization);
int StabilizationClass = core::RegisterObject("Kinematic constraint stabilization").add< Stabilization >();


void Stabilization::correction(SReal* dst, unsigned n) const {
	assert( mstate );
	mstate->copyToBuffer(dst, core::VecCoordId::position(), n);

	// non-zero for stabilized
	unsigned i = 0;
	for(SReal* last = dst + n; dst < last; ++dst, ++i) {
		if( !mask.empty() && !mask[i] ) *dst = 0;
	}
	
}


void Stabilization::dynamics(SReal* dst, unsigned n) const {
	assert( mstate );
	mstate->copyToBuffer(dst, core::VecCoordId::position(), n);

	// zero for stabilized
	unsigned i = 0;
	for(SReal* last = dst + n; dst < last; ++dst, ++i) {
		if( mask.empty() || mask[i] ) *dst = 0;
	}

}




}
}
}
