#include "LDLTResponse.h"

#include <sofa/core/ObjectFactory.h>

namespace sofa {
namespace component {
namespace linearsolver {

SOFA_DECL_CLASS(LDLTResponse);
int LDLTResponseClass = core::RegisterObject("A sparse Cholesky factorization of the response matrix.").add< LDLTResponse >();


void LDLTResponse::factor(const mat& H ) {

    if( _constant.getValue() && !_firstFactorization ) return;
    _firstFactorization = false;

	// TODO make sure no temporary is used ?
	response.compute( H.transpose().selfadjointView<Eigen::Upper>() );
	
	if( response.info() != Eigen::Success ) {
		std::cerr << "warning: non invertible response" << std::endl;
	}

	assert( response.info() == Eigen::Success );
}

void LDLTResponse::solve(cmat& res, const cmat& M) const {
	assert( response.rows() );
	assert( &res != &M );
	res = response.solve( M );
}


void LDLTResponse::solve(vec& res, const vec& x) const {
	assert( response.rows() );
	assert( &res != &x );
	res = response.solve( x );
}

}
}
}
