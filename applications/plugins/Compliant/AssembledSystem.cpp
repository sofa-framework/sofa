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
			
    b = vec::Zero( m );
    v = vec::Zero( m );
				
	if( n ) {
		J.resize(n, m);

		C.resize(n, n);
		phi = vec::Zero(n); 
		lambda = vec::Zero(n); 
		
	}
				
}

unsigned AssembledSystem::size() const { return m + n; }
			

void AssembledSystem::debug(SReal /*thres*/) const {

    std::cerr << "b: " << std::endl
              << b.transpose() << std::endl
	          << "v: " << std::endl
              << v.transpose() << std::endl;
	
	std::cerr << "H:" << std::endl
	          << H << std::endl
	          << "P:" << std::endl
	          << P << std::endl;
	if( n ) { 
		std::cerr << "phi:" << std::endl
		          << phi.transpose() << std::endl;

		std::cerr << "lambda:" << std::endl
		          << lambda.transpose() << std::endl;
		
		std::cerr << "J:" << std::endl 
		          << J << std::endl
		          << "C:" << std::endl
		          << C << std::endl;
	}

}




}
}
}
