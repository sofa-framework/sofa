#include "UnilateralMask.h"
#include <sofa/core/ObjectFactory.h>

#include <sofa/core/behavior/BaseMechanicalState.h>

namespace sofa {
namespace component {
namespace linearsolver {

SOFA_DECL_CLASS(UnilateralMask);
int UnilateralMaskClass = core::RegisterObject("Sparse KKT linear solver").add< UnilateralMask >();


UnilateralMask::UnilateralMask() 
	: mask(initData(&mask, "mask", "inequality mask: 1 for greater than, -1 for lower than")),
	  value(initData(&value, 1.0, "value", "default mask value when mask is not given explicitely"))
{

}


unsigned UnilateralMask::write(SReal* out) const {

	const mask_type& m = mask.getValue();
	
	unsigned n = getContext()->getMechanicalState()->getMatrixSize();
	
	if( m.empty() ) {
		
		for( unsigned i = 0; i < n; ++i) {
			*(out++) = value.getValue();
		}
		
		return n;
	} else {
		assert( m.size() == n );
		
		for( unsigned i = 0, n = m.size(); i < n; ++i) {
			*(out++) = m[i];
		}
		
		return m.size();
	}
	
}

void UnilateralMask::init() {
	// mask_type* m = mask.beginEdit();
	
	// if( m->empty() ) {

		// find dofs/dimension
	core::behavior::BaseMechanicalState* state = getContext()->getMechanicalState();
	if( !state ) throw std::logic_error( getName() + " did not find an mstate");
	
	// 	unsigned size = state->getMatrixSize();
		
	// 	// resize dimension
	// 	m->resize( size, value.getValue() );
	// }
	
	// mask.endEdit();
}

}
}
}
