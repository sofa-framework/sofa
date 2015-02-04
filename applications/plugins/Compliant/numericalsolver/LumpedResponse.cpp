#include "LumpedResponse.h"

#include <sofa/core/ObjectFactory.h>
#include "utils/nan.h"

namespace sofa {
namespace component {
namespace linearsolver {

SOFA_DECL_CLASS(LumpedResponse)
int LumpedResponseClass = core::RegisterObject("A diagonal factorization of the lumped response matrix.").add< LumpedResponse >();



void LumpedResponse::factor( const mat& H, bool semidefinite )
{
	
    diag.resize( H.rows() );

    // lumping
    for( mat::Index k=0 ; k<H.outerSize() ; ++k )
    {
        for( mat::InnerIterator it(H,k) ; it ; ++it )
        {
            diag[it.row()] += it.value();
        }
    }

    if( semidefinite )
    {
        for( mat::Index i=0 ; i<diag.size() ; ++i )
            diag.coeffRef(i) = std::abs(diag.coeff(i)) < std::numeric_limits<real>::epsilon() ? real(0) : real(1) / diag.coeff(i);
    }
    else
    {
        diag = H.diagonal().cwiseInverse();
    }
	
	assert( !has_nan(diag) );
}

void LumpedResponse::solve(cmat& res, const cmat& M) const {
	// TODO make sure this is optimal
	assert( diag.size() == M.rows() );
	assert( &res != &M );
	res = (M.transpose() * diag.asDiagonal() ).transpose();
}


void LumpedResponse::solve(vec& res, const vec& x) const {
	assert( diag.size() == x.size() );
	assert( &res != &x );
	
	res.noalias() = diag.cwiseProduct( x );
}

}
}
}
