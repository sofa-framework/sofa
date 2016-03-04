#ifndef COMPLIANT_EIGENSPARSERESPONSE_INL
#define COMPLIANT_EIGENSPARSERESPONSE_INL

#include "EigenSparseResponse.h"


namespace sofa {
namespace component {
namespace linearsolver {



template<class LinearSolver,bool symmetric>
EigenSparseResponse<LinearSolver,symmetric>::EigenSparseResponse()
    : d_regularize( initData(&d_regularize, std::numeric_limits<real>::epsilon(), "regularize", "add identity*regularize to matrix H to make it definite.") )
    , d_constant( initData(&d_constant, false, "constant", "reuse first factorization") )
    , d_trackSparsityPattern( initData(&d_trackSparsityPattern, false, "trackSparsityPattern", "if the sparsity pattern remains similar from one step to the other, the factorization can be faster") )
    , m_factorized( false )
    , m_previousSize(0)
    , m_previousNonZeros(0)
{}

template<class LinearSolver,bool symmetric>
void EigenSparseResponse<LinearSolver,symmetric>::reinit()
{
    Response::reinit();
    m_factorized = false;
}

template<class LinearSolver,bool symmetric>
void EigenSparseResponse<LinearSolver,symmetric>::factor(const rmat& H) {

#ifndef NDEBUG
    if( !H.rows() ) serr<<"factor - null matrix"<<sendl;
#endif

    if( d_constant.getValue() )
    {
        if( m_factorized ) return;
        else m_factorized = true;
    }

    if( this->f_printLog.getValue() ) serr<<"factor: "<<H<<sendl;


    if( symmetric ) tmp = H.triangularView< Eigen::Lower >(); // only copy the triangular part (default to Lower)
    else tmp = H; // TODO there IS a temporary here, from rmat to cmat. Explicit copy is needed for iterative solvers


    compute( tmp );
	
    if( response.info() != Eigen::Success )
    {
        // try to regularize
        if( d_regularize.getValue() )
        {
            cmat identity(H.rows(),H.cols());
            identity.setIdentity();
            tmp += identity * d_regularize.getValue();
            response.compute( tmp );

            serr << "non invertible matrix, regularizing" << sendl;

            if( response.info() != Eigen::Success )
            {
                serr << "non invertible matrix, even after automatic regularization" << sendl;
//                serr << dmat( tmp ) << sendl;
            }
        }
        else
            serr << "non invertible matrix" << sendl;
	}

    assert( response.info() == Eigen::Success );

}

template<class LinearSolver,bool symmetric>
void EigenSparseResponse<LinearSolver,symmetric>::compute(const cmat& M)
{
    if( !d_trackSparsityPattern.getValue() )
    {
        response.compute( M );
    }
    else
    {
        // TODO the sparsity structure verification is poor
        // but it is enough in some specific cases
        if( M.rows()!=m_previousSize || M.nonZeros()!=m_previousNonZeros )
        {
            response.analyzePattern( M );
            m_previousSize = M.rows();
            m_previousNonZeros = M.nonZeros();
        }
        // If the matrice has the same structure as the previous step,
        // the symbolic decomposition based on the sparcity is still valid.
        response.factorize( M );
    }
}

template<class LinearSolver,bool symmetric>
void EigenSparseResponse<LinearSolver,symmetric>::solve(cmat& res, const cmat& M) const {
	res = response.solve( M );
}


template<class LinearSolver,bool symmetric>
void EigenSparseResponse<LinearSolver,symmetric>::solve(vec& res, const vec& x) const {
	res = response.solve( x );
}


/////////////////////////////////////////////


template<class LinearSolver,bool symmetric>
EigenSparseIterativeResponse<LinearSolver,symmetric>::EigenSparseIterativeResponse()
    : d_iterations( initData(&d_iterations, 100u, "iterations", "max iterations") )
    , d_tolerance( initData(&d_tolerance, (SReal)1e-6, "tolerance", "tolerance") )
{}

template<class LinearSolver,bool symmetric>
void EigenSparseIterativeResponse<LinearSolver,symmetric>::init()
{
    EigenSparseResponse<LinearSolver,symmetric>::init();
    reinit();
}

template<class LinearSolver,bool symmetric>
void EigenSparseIterativeResponse<LinearSolver,symmetric>::reinit()
{
    EigenSparseResponse<LinearSolver,symmetric>::reinit();
    this->response.setMaxIterations( d_iterations.getValue() );
    this->response.setTolerance( d_tolerance.getValue() );
}


}
}
}

#endif
