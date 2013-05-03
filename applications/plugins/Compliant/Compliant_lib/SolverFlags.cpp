#include "SolverFlags.h"
#include <sofa/core/ObjectFactory.h>

#include <sofa/core/behavior/BaseMechanicalState.h>

namespace sofa {
namespace component {
namespace linearsolver {

SOFA_DECL_CLASS(SolverFlags);
int SolverFlagsClass = core::RegisterObject("Arbitrary solver flags for DOFs").add< SolverFlags >();


SolverFlags::SolverFlags() 
	: flags(initData(&flags, "mask", "inequality mask: 1 for greater than, -1 for lower than")),
	  value(initData(&value, unsigned(NO_FLAG), "value", "default mask value when mask is not given explicitly")),
	  data(0)
	  
{
	
}


unsigned SolverFlags::write(value_type* out) const {
	
	const flags_type& f = flags.getValue();
	
	unsigned n = getContext()->getMechanicalState()->getMatrixSize();
	
	// use default value ?
	if( f.empty() ) {
		
		for( unsigned i = 0; i < n; ++i ) {
			*(out++) = value.getValue();
		}
		
		return n;
	} else {
		assert( f.size() == n );
		
		for( unsigned i = 0; i < n; ++i) {
			*(out++) = f[i];
		}
		
		return f.size();
	}
	
}





void SolverFlags::init() {
	
	core::behavior::BaseMechanicalState* state = getContext()->getMechanicalState();
	if( !state ) throw std::logic_error( getName() + " did not find an mstate");
	
}

}
}
}
