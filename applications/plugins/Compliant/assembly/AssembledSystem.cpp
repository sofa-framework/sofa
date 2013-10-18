#include "AssembledSystem.h"
#include <sofa/helper/rmath.h>

#include <iostream>

namespace sofa {
namespace component {
namespace linearsolver {


AssembledSystem::AssembledSystem(unsigned m, unsigned n) 
	: m(m), 
	  n(n),
	  dt(0)
{
	if( !m ) return;

	H.resize(m, m);
	P.resize(m, m);
			
	if( n ) {
		J.resize(n, m);
		C.resize(n, n);
	}
				
}

unsigned AssembledSystem::size() const { return m + n; }
			

void AssembledSystem::debug(SReal /*thres*/) const {
	
	std::cerr << "H:" << std::endl
	          << H << std::endl
	          << "P:" << std::endl
	          << P << std::endl;
	if( n ) { 
			
		std::cerr << "J:" << std::endl 
		          << J << std::endl
		          << "C:" << std::endl
		          << C << std::endl;
	}

}




}
}
}
