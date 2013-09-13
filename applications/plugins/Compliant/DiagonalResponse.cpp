#include "DiagonalResponse.h"

#include <sofa/core/ObjectFactory.h>
#include "utils/nan.h"

namespace sofa {
namespace component {
namespace linearsolver {

SOFA_DECL_CLASS(DiagonalResponse);
int DiagonalResponseClass = core::RegisterObject("A diagonal factorization of the response matrix.").add< DiagonalResponse >();


void DiagonalResponse::factor(const mat& H ) {
	
	diagonal = H.diagonal().cwiseInverse();
	
	assert( !has_nan(diagonal) );
}

void DiagonalResponse::solve(cmat& res, const cmat& M) const {
	// TODO make sure this is optimal
	assert( diagonal.size() == M.rows() );
	assert( &res != &M );
	res = (M.transpose() * diagonal.asDiagonal() ).transpose();
}


void DiagonalResponse::solve(vec& res, const vec& x) const {
	assert( diagonal.size() == x.size() );
	assert( &res != &x );
	
	res.noalias() = diagonal.cwiseProduct( x );
}

}
}
}
