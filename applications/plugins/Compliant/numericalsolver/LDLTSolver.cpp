#include "LDLTSolver.h"
#include "LDLTResponse.h"

#include <sofa/core/ObjectFactory.h>

#include "../utils/sparse.h"

#include <Eigen/SparseCholesky>
#include "SubKKT.h"

using std::cerr;
using std::endl;

namespace sofa {
namespace component {
namespace linearsolver {

SOFA_DECL_CLASS(LDLTSolver)
static int LDLTSolverClass = core::RegisterObject("Direct LDLT solver").add< LDLTSolver >();

typedef AssembledSystem::vec vec;


struct LDLTSolver::pimpl_type {
    typedef Eigen::SimplicialLDLT< cmat >  solver_type;

    solver_type solver;
    cmat PHinvPJT;
    cmat schur;

    SubKKT sub;
};



LDLTSolver::LDLTSolver() 
    : pimpl( new pimpl_type ),
      schur(initData(&schur,
                     true,
                     "schur",
                     "use schur complement"))
{
    
}

LDLTSolver::~LDLTSolver() {
    
}


void LDLTSolver::init() {

    KKTSolver::init();

    // let's find a response
    response = this->getContext()->get<Response>( core::objectmodel::BaseContext::Local );

    // fallback in case we missed
    if( !response ) {
        response = new LDLTResponse();
        this->getContext()->addObject( response );
        
        serr << "fallback response class: "
             << response->getClassName()
             << " added to the scene" << sendl;
    }
}


// TODO move to PIMPL
static LDLTSolver::cmat tmp;

void LDLTSolver::factor_schur( const AssembledSystem& sys ) {

     // schur complement
    SubKKT::projected_primal(pimpl->sub, sys);
    pimpl->sub.factor(*response);

    // much cleaner now ;-)
    if( sys.n ) {
        {
            scoped::timer step("schur assembly");
            pimpl->sub.solve_opt(*response, pimpl->PHinvPJT, sys.J );

            // TODO hide this somewhere
            tmp = sys.J;
            pimpl->schur = sys.C.transpose();
            
            sparse::fast_add_prod(pimpl->schur, tmp, pimpl->PHinvPJT);
            
            // pimpl->schur = sys.C.transpose() + (sys.J * pimpl->HinvPJT).triangularView<Eigen::Lower>();
        }
    
        if( debug.getValue() ){
            typedef AssembledSystem::dmat dmat;
            serr << "factor, PHinvPJT = " << sendl << dmat(pimpl->PHinvPJT) << sendl
                 << "factor, schur = " << sendl << dmat(pimpl->schur) << sendl;
        }

        {
            scoped::timer step("schur factorization");
            pimpl->solver.compute( pimpl->schur );
        }

        if( pimpl->solver.info() == Eigen::NumericalIssue ){
            serr << "factor: schur is not psd. System solution will be wrong." << sendl
                 << pimpl->schur << sendl;
        }
    }
}


void LDLTSolver::factor(const AssembledSystem& sys) {

    if( schur.getValue() ) {
        factor_schur( sys );
    } else {
        SubKKT::projected_kkt(pimpl->sub, sys);
        pimpl->sub.factor(*response);
    }
}

void LDLTSolver::solve(vec& res,
                       const AssembledSystem& sys,
                       const vec& rhs) const {
    if( schur.getValue() ) {
        solve_schur(res, sys, rhs);
    } else {
        solve_kkt(res, sys, rhs);
    }

}


void LDLTSolver::solve_kkt(vec& res,
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

void LDLTSolver::solve_schur(vec& res,
                             const AssembledSystem& sys,
                             const vec& rhs) const {

    assert( res.size() == sys.size() );
    assert( rhs.size() == sys.size() );

    vec free;

    if( debug.getValue() ){
        serr << "solve, rhs = " << rhs.transpose() << sendl
             << "solve, H = " << endl << dmat(sys.H) << sendl;
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


}
}
}

