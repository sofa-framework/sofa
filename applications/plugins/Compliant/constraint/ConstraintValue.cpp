#include "ConstraintValue.h"

#include <sofa/core/ObjectFactory.h>
#include "../utils/map.h"
using std::cerr;
using std::endl;

namespace sofa {
namespace component {
namespace odesolver {


SOFA_DECL_CLASS(ConstraintValue)
int ConstaintValueClass = core::RegisterObject("Constraint value abstraction").add< ConstraintValue >();


ConstraintValue::ConstraintValue( mstate_type* mstate )
    : BaseConstraintValue( mstate )
{
}

void ConstraintValue::correction(SReal* dst, unsigned n, unsigned dim, const core::MultiVecCoordId&, const core::MultiVecDerivId&) const {
	
//	for(SReal* last = dst + n*dim; dst < last; ++dst) {
//		*dst = 0;
//	}
	
    memset( dst, 0, n*dim*sizeof(SReal) );
}


void ConstraintValue::dynamics(SReal* dst, unsigned n, unsigned dim, bool, const core::MultiVecCoordId& posId, const core::MultiVecDerivId&) const {
    assert( mstate );

    unsigned size = n*dim;

    mstate->copyToBuffer(dst, posId.getId(mstate.get()), size);

    using namespace utils;
    map(dst, size) *= (-1.0/this->getContext()->getDt());
}



}
}
}
