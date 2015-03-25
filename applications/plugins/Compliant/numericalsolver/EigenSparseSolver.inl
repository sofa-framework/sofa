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


    // for kkt solve, pimpl will have the same API as a Response
    // (such as the same functions can be called in SubKKT)

    void factor(const rmat& A)
    {
        if( symmetric ) tmp = A.template triangularView< Eigen::Lower >();  // only copy the triangular part (default to Lower)
        else tmp = A;    // TODO there IS a temporary here, from rmat to cmat

        solver.compute( tmp ); // the conversion from rmat to cmat needs to be explicit for iterative solvers

        if( solver.info() != Eigen::Success ) {
            std::cerr << "EigenSparseSolver non invertible matrix" << std::endl;
        }
    }

    void solve(vec& lval, const vec& rval) const
    {
        lval = solver.solve( rval );
    }
};






template<class LinearSolver,bool symmetric>
EigenSparseSolver<LinearSolver,symmetric>::EigenSparseSolver()
    : schur(initData(&schur,
                     true,
                     "schur",
                     "use schur complement"))
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

    if( schur.getValue() && !response )
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
}

template<class LinearSolver,bool symmetric>
void EigenSparseSolver<LinearSolver,symmetric>::factor(const AssembledSystem& sys) {

    if( schur.getValue() ) {
        factor_schur( sys );
    } else {
        SubKKT::projected_kkt(pimpl->sub, sys, 1e-14, symmetric);
        pimpl->sub.factor( *pimpl );
    }
}

template<class LinearSolver,bool symmetric>
void EigenSparseSolver<LinearSolver,symmetric>::solve(vec& res,
                       const AssembledSystem& sys,
                       const vec& rhs) const {
    if( schur.getValue() ) {
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

    // much cleaner now ;-)
    if( sys.n ) {
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
                pimpl->solver.compute( pimpl->tmp );
            }
            else
                pimpl->solver.compute( pimpl->schur );
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

