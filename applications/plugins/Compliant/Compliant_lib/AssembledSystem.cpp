#include "AssembledSystem.h"
#include <helper.h>

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
		lambda = vec::Zero(n); 
		
	}
				
}

unsigned AssembledSystem::size() const { return m + n; }
			

void AssembledSystem::debug(SReal /*thres*/) const {

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

		std::cerr << "lambda:" << std::endl
		          << lambda.transpose() << std::endl;
		
		std::cerr << "J:" << std::endl 
		          << J << std::endl
		          << "C:" << std::endl
		          << C << std::endl;
	}

}


bool AssembledSystem::isDiagonalDominant() const
{
    typedef typename mat::Index Index;

    mat PH = P*H;

    if( n )
    {
        mat PJt = P*J.transpose();

        for( unsigned i=0 ; i<m ; ++i )
        {
            real d = helper::rabs(PH.coeff(i,i));

            real o = -d;
            for( Index j=0 ; j<PH.cols()  ; ++j ) o += helper::rabs(PH.coeff(i,j));
            for( Index j=0 ; j<PJt.cols() ; ++j ) o += helper::rabs(PJt.coeff(i,j));

            if( o > d ) return false;
        }

        for( unsigned i=0 ; i<n ; ++i )
        {
            real d = helper::rabs(C.coeff(i,i));

            real o = -d;
            for( Index j=0 ; j<C.cols() ; ++j ) o += helper::rabs(C.coeff(i,j));
            for( Index j=0 ; j<J.cols() ; ++j ) o += helper::rabs(J.coeff(i,j));

            if( o > d ) return false;
        }
    }
    else
    {
        for( unsigned i=0 ; i<m ; ++i )
        {
            real d = helper::rabs(PH.coeff(i,i));

            real o = -d;
            for( Index j=0 ; j<PH.cols() ; ++j ) o += helper::rabs(PH.coeff(i,j));

            if( o > d ) return false;
        }
    }

    return true;
}

}
}
}
