#ifndef COMPLIANT_EIGENSPARSERESPONSE_INL
#define COMPLIANT_EIGENSPARSERESPONSE_INL

#include "EigenSparseResponse.h"


namespace sofa {
namespace component {
namespace linearsolver {



template<class LinearSolver, int symmetric>
EigenSparseResponse<LinearSolver,symmetric>::EigenSparseResponse()
    : d_regularize( initData(&d_regularize, std::numeric_limits<real>::epsilon(), "regularize", "add identity*regularize to matrix H to make it definite."))
    , d_constant( initData(&d_constant, false, "constant", "reuse first factorization"))
    , m_factorized( false )
{}

template<class LinearSolver, int symmetric>
void EigenSparseResponse<LinearSolver,symmetric>::reinit()
{
    Response::reinit();
    m_factorized = false;
}

template<class LinearSolver, int symmetric>
void EigenSparseResponse<LinearSolver,symmetric>::factor(const mat& H, bool semidefinite ) {

#ifndef NDEBUG
    if( !H.rows() ) serr<<"factor - null matrix"<<sendl;
#endif

    if( d_constant.getValue() && m_factorized ) return;

    m_factorized = true;

    if( d_regularize.getValue() && semidefinite ) {
		// add a tiny diagonal matrix to make H psd.
        // TODO add epsilon only on the empty diagonal entries?
        system_type::rmat identity(H.rows(),H.cols());
        identity.setIdentity();
        if( symmetric ) response.compute( (H + identity * d_regularize.getValue()).eval().template triangularView< symmetric >() ); // TODO there IS a temporary hereonly copy the triangular part (default to Lower)
        else response.compute( H + identity * d_regularize.getValue() ); // TODO there IS a temporary here, from rmat to cmat
    }
    else
    {
        /*if( symmetric ) response.compute( H.triangularView< symmetric >() ); // only copy the triangular part (default to Lower)
        else*/ response.compute( H ); // TODO there IS a temporary here, from rmat to cmat
    }

	
	if( response.info() != Eigen::Success ) {
        serr << "non invertible matrix" << sendl;
	}

    assert( response.info() == Eigen::Success );

}

template<class LinearSolver, int symmetric>
void EigenSparseResponse<LinearSolver,symmetric>::solve(cmat& res, const cmat& M) const {
	res = response.solve( M );
}


template<class LinearSolver, int symmetric>
void EigenSparseResponse<LinearSolver,symmetric>::solve(vec& res, const vec& x) const {
	res = response.solve( x );
}



}
}
}

#endif
