#include "DiagonalResponse.h"

#include <sofa/core/ObjectFactory.h>
#include "utils/nan.h"

namespace sofa {
namespace component {
namespace linearsolver {

SOFA_DECL_CLASS(DiagonalResponse);
int DiagonalResponseClass = core::RegisterObject("A diagonal factorization of the response matrix.").add< DiagonalResponse >();


DiagonalResponse::DiagonalResponse() 
    : constant(initData(&constant, false, "constant", "reuse first factorization"))
{

}

void DiagonalResponse::factor( const mat& H, bool semidefinite ) {
	
    if( constant.getValue() && diag.size() == H.rows() ) return;

    if( semidefinite )
    {
        diag = H.diagonal();
        for( unsigned i=0 ; i<H.rows() ; ++i )
            diag.coeffRef(i) = std::abs(diag.coeff(i)) < std::numeric_limits<real>::epsilon() ? real(0) : real(1) / diag.coeff(i);
    }
    else
    {
        diag = H.diagonal().cwiseInverse();
    }
	
	assert( !has_nan(diag) );
}

void DiagonalResponse::solve(cmat& res, const cmat& M) const {
	// TODO make sure this is optimal
	assert( diag.size() == M.rows() );
	assert( &res != &M );
	res = (M.transpose() * diag.asDiagonal() ).transpose();
}


void DiagonalResponse::solve(vec& res, const vec& x) const {
	assert( diag.size() == x.size() );
	assert( &res != &x );
	
	res.noalias() = diag.cwiseProduct( x );
}

}
}
}
