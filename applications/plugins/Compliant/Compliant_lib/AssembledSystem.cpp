#include "AssembledSystem.h"

namespace sofa {
namespace component {
namespace linearsolver {
		
AssembledSystem::block::block(unsigned offset, unsigned size) 
	: offset(offset),
	  size(size),
	  data(0) {

}

AssembledSystem::AssembledSystem(unsigned m, unsigned n) 
	: m(m), 
	  n(n),
	  dt(0),
	  blocks()
{
	if( !m ) return;
				
	// M.resize(m, m);
	// K.resize(m, m);

	H.resize(m, m);
	P.resize(m, m);
			
	f = vec::Zero( m );
	v = vec::Zero( m );
	p = vec::Zero( m );
				
	if( n ) {
		J.resize(n, m);

		C.resize(n, n);
		phi = vec::Zero(n); 

		flags.setConstant(n, SolverFlags::NO_FLAG);
	}
				
}

unsigned AssembledSystem::size() const { return m + n; }
			
}
}
}
