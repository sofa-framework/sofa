#include "ConstraintValue.h"

#include <sofa/core/ObjectFactory.h>

namespace sofa {
namespace component {
namespace odesolver {


SOFA_DECL_CLASS(ConstraintValue);
int ConstaintValueClass = core::RegisterObject("Constraint value abstraction").add< ConstraintValue >();


void ConstraintValue::init() {
	mstate = this->getContext()->get<mstate_type>(core::objectmodel::BaseContext::Local);
}

void ConstraintValue::correction(SReal* dst, unsigned n) const {
	
	for(SReal* last = dst + n; dst < last; ++dst) {
		*dst = 0;
	}
	
}


void ConstraintValue::dynamics(SReal* dst, unsigned n) const {
	assert( mstate );
	
	mstate->copyToBuffer(dst, core::VecCoordId::position(), n);
}



}
}
}
