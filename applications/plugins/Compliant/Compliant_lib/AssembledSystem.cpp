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
			

void AssembledSystem::debug(SReal thres) const {

	std::cerr << "f: " << std::endl
	          << f.transpose() << std::endl
	          << "v: " << std::endl
	          << v.transpose() << std::endl
	          << "p:" << std::endl
	          << p.transpose() << std::endl;
	
	std::cerr << "H:" << std::endl
	          << H << std::endl
	          << "P:" << std::endl
	          << P << std::endl;
	if( n ) { 
		std::cerr << "phi:" << std::endl
		          << phi.transpose() << std::endl;
		
		std::cerr << "J:" << std::endl 
		          << J << std::endl
		          << "C:" << std::endl
		          << C << std::endl;
	}

}


}
}
}
