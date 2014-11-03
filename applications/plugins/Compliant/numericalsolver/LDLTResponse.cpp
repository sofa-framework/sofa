#include "LDLTResponse.h"

#include <sofa/core/ObjectFactory.h>

namespace sofa {
namespace component {
namespace linearsolver {

SOFA_DECL_CLASS(LDLTResponse);
int LDLTResponseClass = core::RegisterObject("A sparse Cholesky factorization of the response matrix.").add< LDLTResponse >();
 

LDLTResponse::LDLTResponse()
    : regularize( initData(&regularize, 
                           std::numeric_limits<real>::epsilon(),
						   "regularize", 
						   "add identity*regularize to matrix H to make it definite.")),
	  constant( initData(&constant, 
						 false,
						 "constant",
                         "reuse first factorization")),
    factorized( false )
{}


void LDLTResponse::reinit()
{
    Response::reinit();
    factorized = false;
}

void LDLTResponse::factor(const mat& H, bool semidefinite ) {

    if( constant.getValue() && factorized ) return;

    factorized = true;

    if( regularize.getValue() && semidefinite ) {
		// add a tiny diagonal matrix to make H psd.
        // TODO add epsilon only on the empty diagonal entries?
        system_type::rmat identity(H.rows(),H.cols());
        identity.setIdentity();
        response.compute( ( H + identity * regularize.getValue() ).selfadjointView<Eigen::Upper>() );
    }
    else
    {
        // TODO make sure no temporary is used ?
        response.compute( H.selfadjointView<Eigen::Upper>() );
    }

	
	if( response.info() != Eigen::Success ) {
        serr << "non invertible response" << sendl;
	}

	assert( response.info() == Eigen::Success );

}

void LDLTResponse::solve(cmat& res, const cmat& M) const {
	assert( response.rows() );
	res = response.solve( M );
}


void LDLTResponse::solve(vec& res, const vec& x) const {
	assert( response.rows() );
	res = response.solve( x );
}

}
}
}
