#include "ConstraintValue.h"

#include <sofa/core/ObjectFactory.h>
#include "../utils/map.h"
using std::cerr;
using std::endl;

namespace sofa {
namespace component {
namespace odesolver {


SOFA_DECL_CLASS(ConstraintValue);
int ConstaintValueClass = core::RegisterObject("Constraint value abstraction").add< ConstraintValue >();


ConstraintValue::ConstraintValue( mstate_type* mstate )
    : BaseConstraintValue( mstate )
//    , dampingRatio( initData(&dampingRatio, SReal(0.0), "dampingRatio", "Weight of the velocity in the constraint violation"))
{
}

void ConstraintValue::correction(SReal* dst, unsigned n) const {
	
	for(SReal* last = dst + n; dst < last; ++dst) {
		*dst = 0;
	}
	
}


void ConstraintValue::dynamics(SReal* dst, unsigned n) const {
    assert( mstate );

	mstate->copyToBuffer(dst, core::VecCoordId::position(), n);

	using namespace utils;
    map(dst, n) = -map(dst, n) / this->getContext()->getDt();

    // TODO damping
	
}



}
}
}
