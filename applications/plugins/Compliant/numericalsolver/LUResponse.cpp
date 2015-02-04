#include "LUResponse.h"

#include <sofa/core/ObjectFactory.h>

namespace sofa {
namespace component {
namespace linearsolver {

SOFA_DECL_CLASS(LUResponse)
static int LUResponseClass = core::RegisterObject("A sparse LU factorization of the response matrix.").add< LUResponse >();
 

LUResponse::LUResponse()
    : regularize( initData(&regularize, 
                           std::numeric_limits<real>::epsilon(),
						   "regularize", 
                           "add identity*regularize to matrix H to make it definite."))
{}



void LUResponse::factor(const mat& H, bool semidefinite ) {

    if( regularize.getValue() && semidefinite ) {
		// add a tiny diagonal matrix to make H psd.
        // TODO add epsilon only on the empty diagonal entries?
        system_type::rmat identity(H.rows(), H.cols());
        identity.setIdentity();
        response.compute( ( H + identity * regularize.getValue() ) );
    }
    else
    {
        // TODO make sure no temporary is used ?
        response.compute( H );
    }

	
	if( response.info() != Eigen::Success ) {
        serr << "non-invertible response" << sendl;
	}

	assert( response.info() == Eigen::Success );

}

void LUResponse::solve(cmat& res, const cmat& M) const {
	assert( response.rows() );
	res = response.solve( M );
}


void LUResponse::solve(vec& res, const vec& x) const {
	assert( response.rows() );
	res = response.solve( x );
}

}
}
}
