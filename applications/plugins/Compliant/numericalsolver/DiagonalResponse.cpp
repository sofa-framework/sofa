#include "DiagonalResponse.h"

#include <sofa/core/ObjectFactory.h>
#include "../utils/nan.h"

namespace sofa {
namespace component {
namespace linearsolver {

SOFA_DECL_CLASS(DiagonalResponse)
int DiagonalResponseClass = core::RegisterObject("A diagonal factorization of the response matrix.").add< DiagonalResponse >();


DiagonalResponse::DiagonalResponse() 
    : constant(initData(&constant, false, "constant", "reuse first factorization"))
    , factorized(false)
{

}


void DiagonalResponse::reinit()
{
    Response::reinit();
    factorized = false;
}


void DiagonalResponse::factor(const rmat& H) {
	
    if( constant.getValue() )
    {
        if( factorized ) return;
        else factorized = true;
    }

    diag = H.diagonal().cwiseInverse();

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
