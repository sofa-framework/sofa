#include "LDLTSolver.h"
#include "LDLTResponse.h"

#include <sofa/core/ObjectFactory.h>

#include "../utils/scoped.h"
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
    cmat HinvPJT;
    cmat schur;

    SubKKT sub;
};



LDLTSolver::LDLTSolver() 
    : pimpl( new pimpl_type ) {
    
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



void LDLTSolver::factor_schur( const cmat& schur )
{
    if( debug.getValue() ){
        typedef AssembledSystem::dmat dmat;
        serr << "factor, HinvPJT = " << sendl << dmat(pimpl->HinvPJT) << sendl
             << "factor, schur = " << sendl << dmat(schur) << sendl;
    }

    {
        scoped::timer step("schur factorization");
        pimpl->solver.compute( schur );
    }

    if( pimpl->solver.info() == Eigen::NumericalIssue ){
        serr << "factor: schur is not psd. System solution will be wrong." << sendl
             << schur << sendl;
    }
}

static LDLTSolver::cmat tmp;

void LDLTSolver::factor(const AssembledSystem& sys) {

    // response matrix
    assert( response );

    SubKKT::projected_primal(pimpl->sub, sys);
    pimpl->sub.factor(*response);

    // much cleaner now ;-)
    if( sys.n ) {
        {
            scoped::timer step("schur assembly");
            // sub.solve(*response, pimpl->HinvPJT, sys.J.transpose() );
            pimpl->sub.solve_opt(*response, pimpl->HinvPJT, sys.J );

            tmp = sys.J;
            pimpl->schur = sys.C.transpose();
            sparse::fast_add_prod(pimpl->schur, tmp, pimpl->HinvPJT);
            
            // pimpl->schur = sys.C.transpose() + (sys.J * pimpl->HinvPJT).triangularView<Eigen::Lower>();
        }
        factor_schur( pimpl->schur );
    }

}


void LDLTSolver::solve(AssembledSystem::vec& res,
                       const AssembledSystem& sys,
                       const AssembledSystem::vec& rhs) const {

    assert( res.size() == sys.size() );
    assert( rhs.size() == sys.size() );

    vec free;
    
    typedef AssembledSystem::dmat dmat;

    if( debug.getValue() ){
        serr << "solve, rhs = " << rhs.transpose() << sendl
             << "solve, free = " << free.transpose() << sendl
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
        vec tmp = rhs.tail( sys.n ) - pimpl->HinvPJT.transpose() * rhs.head( sys.m );
        
        // lambdas
        res.tail( sys.n ) = pimpl->solver.solve( tmp );
        
        // constraint forces
        res.head( sys.m ) += pimpl->HinvPJT * res.tail( sys.n );
        
        if( debug.getValue() ){
            serr << "solve, free motion constraint error= "
                 << -tmp.transpose() << sendl
                
                 << "solve, lambda = "
                 << res.tail(sys.n).transpose() << sendl
                
                 << "solve, constraint forces = "
                 << (pimpl->HinvPJT * res.tail( sys.n)).transpose() << sendl;
        }
    }

} 


}
}
}

