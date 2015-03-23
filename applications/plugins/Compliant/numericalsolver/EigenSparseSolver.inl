#ifndef COMPLIANT_EIGENSPARSESOLVER_INL
#define COMPLIANT_EIGENSPARSESOLVER_INL

#include "EigenSparseSolver.h"
#include "SubKKT.h"

#include "EigenSparseResponse.h"

#include "../utils/sparse.h"


namespace sofa {
namespace component {
namespace linearsolver {


typedef AssembledSystem::vec vec;

template<class LinearSolver,bool symmetric>
struct EigenSparseSolver<LinearSolver,symmetric>::pimpl_type {
    typedef LinearSolver solver_type;

    solver_type solver;
    cmat PHinvPJT;
    cmat schur;

    cmat tmp;
    
    SubKKT sub;
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

template<class LinearSolver,bool symmetric>
void EigenSparseSolver<LinearSolver,symmetric>::factor(const AssembledSystem& sys) {

    if( schur.getValue() ) {
        factor_schur( sys );
    } else {
        SubKKT::projected_kkt(pimpl->sub, sys, 1e-14, symmetric);
        pimpl->sub.factor(*response);
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
            // taking only a triangular part of the Schur complement only works for Direct solvers (not for iterative ones)
            // and anyway it seems slower!
//            if( symmetric )
//            {
//                const cmat& schur = pimpl->schur;
////                pimpl->solver.compute( schur.selfadjointView<Eigen::Upper>() );
////                pimpl->solver.compute( schur.triangularView<Eigen::Lower>() );
////                 pimpl->solver.compute( schur.triangularView< LinearSolver::UpLo >() ); // need specializations to get UpLo on a generic way
//            }
//            else
//                pimpl->solver.compute( pimpl->schur );

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

    pimpl->sub.solve(*response, res, tmp);
    
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
    this->pimpl->solver.setMaxIterations( d_iterations.getValue() );
    this->pimpl->solver.setTolerance( d_tolerance.getValue() );
}

}
}
}

#endif

