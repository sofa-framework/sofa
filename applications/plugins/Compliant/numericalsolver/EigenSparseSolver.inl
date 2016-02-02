#ifndef COMPLIANT_EIGENSPARSESOLVER_INL
#define COMPLIANT_EIGENSPARSESOLVER_INL

#include "EigenSparseSolver.h"
#include "SubKKT.inl"

#include "EigenSparseResponse.h"

#include "../utils/sparse.h"


namespace sofa {
namespace component {
namespace linearsolver {


typedef AssembledSystem::vec vec;

template<class LinearSolver,bool symmetric>
struct EigenSparseSolver<LinearSolver,symmetric>::pimpl_type {
    typedef LinearSolver solver_type;

    mutable solver_type solver;
    cmat PHinvPJT;
    cmat schur;

    cmat tmp;
    
    SubKKT sub;

    bool m_trackSparsityPattern;
    cmat::Index m_previousSize;
    cmat::Index m_previousNonZeros;

    pimpl_type()
        : m_trackSparsityPattern(false)
        , m_previousSize(0)
        , m_previousNonZeros(0)
    {}


    // for kkt solve, pimpl will have the same API as a Response
    // (such as the same functions can be called in SubKKT)

    void factor(const rmat& A)
    {
        if( symmetric ) tmp = A.template triangularView< Eigen::Lower >();  // only copy the triangular part (default to Lower)
        else tmp = A;    // TODO there IS a temporary here, from rmat to cmat

        // the conversion from rmat to cmat needs to be explicit for iterative solvers
        compute( tmp );

        if( solver.info() != Eigen::Success ) {
            std::cerr << "EigenSparseSolver non invertible matrix" << std::endl;
        }
    }

    void solve(vec& lval, const vec& rval) const
    {
        lval = solver.solve( rval );
    }

    void compute(const cmat& A)
    {
        if( !m_trackSparsityPattern )
            solver.compute( A );
        else
        {
            // TODO the sparsity structure verification is poor
            // but it is enough in some specific cases
            if( tmp.rows()!=m_previousSize || tmp.nonZeros()!=m_previousNonZeros )
            {
                solver.analyzePattern( A );
                m_previousSize = A.rows();
                m_previousNonZeros = A.nonZeros();
            }
            solver.factorize( A );
        }
    }

};






template<class LinearSolver,bool symmetric>
EigenSparseSolver<LinearSolver,symmetric>::EigenSparseSolver()
    : d_schur(initData(&d_schur,
                     true,
                     "schur",
                     "use schur complement"))
    , d_regularization(initData(&d_regularization, std::numeric_limits<SReal>::epsilon(), "regularization", "Optional diagonal Tikhonov regularization on constraints"))
    , d_trackSparsityPattern( initData(&d_trackSparsityPattern, false, "trackSparsityPattern", "if the sparsity pattern remains similar from one step to the other, the factorization can be faster") )
    , pimpl( new pimpl_type )
{}

template<class LinearSolver,bool symmetric>
EigenSparseSolver<LinearSolver,symmetric>::~EigenSparseSolver()
{}

template<class LinearSolver,bool symmetric>
void EigenSparseSolver<LinearSolver,symmetric>::init() {

    KKTSolver::init();
    reinit();
}

template<class LinearSolver,bool symmetric>
void EigenSparseSolver<LinearSolver,symmetric>::reinit() {

    KKTSolver::reinit();

    if( d_schur.getValue() && !response )
    {
        // let's find a response
        response = this->getContext()->template get<Response>( core::objectmodel::BaseContext::Local );

        // fallback in case we missed
        if( !response ) {

            if( symmetric ) response = new LDLTResponse();
            else response = new LUResponse();
            this->getContext()->addObject( response );

            serr << "fallback response class: "
                 << response->getClassName()
                 << " added to the scene" << sendl;
        }
    }

    pimpl->m_trackSparsityPattern = d_trackSparsityPattern.getValue();
}

template<class LinearSolver,bool symmetric>
void EigenSparseSolver<LinearSolver,symmetric>::factor(const AssembledSystem& sys) {

    if( d_schur.getValue() ) {
        factor_schur( sys );
    } else {

        SubKKT::projected_kkt(pimpl->sub, sys, d_regularization.getValue(), symmetric);

        pimpl->sub.factor( *pimpl );

        if( debug.getValue() )
            serr<<"H: "<<pimpl->sub.A<<sendl;
    }
}

template<class LinearSolver,bool symmetric>
void EigenSparseSolver<LinearSolver,symmetric>::solve(vec& res,
                       const AssembledSystem& sys,
                       const vec& rhs) const {
    if( d_schur.getValue() ) {
        solve_schur(res, sys, rhs);
    } else {
        solve_kkt(res, sys, rhs);
    }

}

template<class LinearSolver,bool symmetric>
void EigenSparseSolver<LinearSolver,symmetric>::factor_schur( const AssembledSystem& sys ) {

    // schur complement
    SubKKT::projected_primal(pimpl->sub, sys);
    pimpl->sub.factor(*response);

    if( sys.n )
    {
        {
            scoped::timer step("schur assembly");

            pimpl->sub.solve_opt(*response, pimpl->PHinvPJT, sys.J );

            // TODO hide this somewhere
            pimpl->tmp = sys.J;
            pimpl->schur = sys.C.transpose();
            
            sparse::fast_add_prod(pimpl->schur, pimpl->tmp, pimpl->PHinvPJT);
        }
    
        if( debug.getValue() ){
            typedef AssembledSystem::dmat dmat;
            serr << "factor, PHinvPJT = " << sendl << dmat(pimpl->PHinvPJT) << sendl
                 << "factor, schur = " << sendl << dmat(pimpl->schur) << sendl;
        }

        {
            scoped::timer step("schur factorization");

            if( symmetric )
            {
                pimpl->tmp = pimpl->schur.template triangularView< Eigen::Lower >(); // to make it work with MINRES
                pimpl->compute( pimpl->tmp );
            }
            else
                pimpl->compute( pimpl->schur );
        }

        if( pimpl->solver.info() == Eigen::NumericalIssue ){
            serr << "factor: schur factorization failed :-/ ";
            if( symmetric ) serr<< "(is schur psd ?)";
            serr << sendl << pimpl->schur << sendl;
        }
    }
}


template<class LinearSolver,bool symmetric>
void EigenSparseSolver<LinearSolver,symmetric>::solve_kkt(vec& res,
                           const AssembledSystem& sys,
                           const vec& rhs) const {
    assert( res.size() == sys.size() );
    assert( rhs.size() == sys.size() );

    vec tmp = vec::Zero(sys.size());

    // flip dual value to agree with symmetric kkt
    tmp.head(sys.m) = rhs.head(sys.m);
    if( sys.n ) tmp.tail(sys.n) = -rhs.tail(sys.n);

    pimpl->sub.solve( *pimpl, res, tmp );
    
}

template<class LinearSolver,bool symmetric>
void EigenSparseSolver<LinearSolver,symmetric>::solve_schur(vec& res,
                             const AssembledSystem& sys,
                             const vec& rhs) const {

    assert( res.size() == sys.size() );
    assert( rhs.size() == sys.size() );

    vec free;

    if( debug.getValue() ){
        serr << "solve, rhs = " << rhs.transpose() << sendl
             << "solve, H = " << sendl << dmat(sys.H) << sendl;
    }

    // in place solve
    pimpl->sub.solve(*response, free, rhs.head(sys.m));
    
    if( debug.getValue() ){
        serr << "solve, free motion solution = " << free.transpose() << sendl
            // << "solve, verification = " << (sys.H * free).transpose() << sendl;
             << "solve, sys.m = " << sys.m
             << ", sys.n = " << sys.n
             << ", rhs.size = " << rhs.size() << sendl;

    }
    
    res.head( sys.m ) = free;
    
    if( sys.n ) {

        vec tmp = rhs.tail( sys.n ) - pimpl->PHinvPJT.transpose() * rhs.head( sys.m );
        
        // lambdas
        res.tail( sys.n ) = pimpl->solver.solve( tmp );
        
        // constraint forces
        res.head( sys.m ) += pimpl->PHinvPJT * res.tail( sys.n );
        
        if( debug.getValue() ){
            serr << "solve, free motion constraint error= "
                 << -tmp.transpose() << sendl
                
                 << "solve, lambda = "
                 << res.tail(sys.n).transpose() << sendl
                
                 << "solve, constraint forces = "
                 << (pimpl->PHinvPJT * res.tail( sys.n)).transpose() << sendl;
        }
    }

} 



/////////////////////////////////////////////


template<class LinearSolver,bool symmetric>
EigenSparseIterativeSolver<LinearSolver,symmetric>::EigenSparseIterativeSolver()
    : d_iterations( initData(&d_iterations, 100u, "iterations", "max iterations") )
    , d_tolerance( initData(&d_tolerance, (SReal)1e-6, "tolerance", "tolerance") )
{}

template<class LinearSolver,bool symmetric>
void EigenSparseIterativeSolver<LinearSolver,symmetric>::init()
{
    EigenSparseSolver<LinearSolver,symmetric>::init();
    reinit();
}

template<class LinearSolver,bool symmetric>
void EigenSparseIterativeSolver<LinearSolver,symmetric>::reinit()
{
    EigenSparseSolver<LinearSolver,symmetric>::reinit();
    this->pimpl->solver.setMaxIterations( d_iterations.getValue() );
    this->pimpl->solver.setTolerance( d_tolerance.getValue() );
}

}
}
}

#endif

