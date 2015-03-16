#include "LUSolver.h"
#include "LUResponse.h"

#include <sofa/core/ObjectFactory.h>

#include "../utils/sparse.h"
#include <Eigen/SparseLU>
#include "SubKKT.h"


using std::cerr;
using std::endl;

namespace sofa {
namespace component {
namespace linearsolver {

SOFA_DECL_CLASS(LUSolver)
static int LUSolverClass = core::RegisterObject("Direct LU solver").add< LUSolver >();

typedef AssembledSystem::vec vec;


struct LUSolver::pimpl_type {
    
    typedef Eigen::SparseLU< cmat > solver_type;

    solver_type solver;
    cmat PHinvPJT;
    cmat schur;

    SubKKT sub;
};



LUSolver::LUSolver()
    : pimpl( new pimpl_type ){

}

LUSolver::~LUSolver() {

}


void LUSolver::init() {

    KKTSolver::init();

    // let's find a response
    response = this->getContext()->get<Response>( core::objectmodel::BaseContext::Local );

    // fallback in case we missed
    if( !response ) {
        response = new LUResponse();
        this->getContext()->addObject( response );

        serr << "fallback response class: "
             << response->getClassName()
             << " added to the scene" << sendl;
    }
}



void LUSolver::factor_schur( const cmat& schur )
{
    if( debug.getValue() ){
        typedef AssembledSystem::dmat dmat;
        serr << "factor, PHinvPJT = " << sendl
             << dmat(pimpl->PHinvPJT) << sendl
             << "factor, schur = " << sendl << dmat(schur)
             << sendl;
    }

    {
        scoped::timer step("Schur factorization");
        pimpl->solver.compute( schur );
    }

    if( pimpl->solver.info() == Eigen::NumericalIssue ){
        serr << "factor: schur factorization failed." << sendl
             << schur << sendl;
    }
}


// TODO move to PIMPL
static LUSolver::cmat tmp;

void LUSolver::factor(const AssembledSystem& sys) {

    // response matrix
    assert( response );

    SubKKT::projected_primal(pimpl->sub, sys);
    pimpl->sub.factor(*response);

    if( sys.n ) {
        {
            scoped::timer step("schur assembly");
            pimpl->sub.solve_opt(*response, pimpl->PHinvPJT, sys.J );

            // TODO hide this
            tmp = sys.J;
            pimpl->schur = sys.C.transpose();
            
            sparse::fast_add_prod(pimpl->schur, tmp, pimpl->PHinvPJT);
            
            // pimpl->schur = sys.C.transpose() + (sys.J * pimpl->HinvPJT).triangularView<Eigen::Lower>();
        }
        factor_schur( pimpl->schur );
    }
    
}


void LUSolver::solve(vec& res,
                     const AssembledSystem& sys,
                     const vec& rhs) const {

    assert( res.size() == sys.size() );
    assert( rhs.size() == sys.size() );

    vec free;


    if( debug.getValue() ){
        serr << "solve, rhs = " << rhs.transpose() << sendl
             << "solve, H = " << sendl << dmat(sys.H) << sendl;
    }

    pimpl->sub.solve(*response, free, rhs.head(sys.m));    

    if( debug.getValue() ){
        serr << "solve, free motion solution = " << free.transpose() << sendl
             << "solve, verification = " << (sys.H * free).transpose() << sendl
             << "solve, sys.m = " << sys.m
             << ", sys.n = " << sys.n
             << ", rhs.size = " << rhs.size() << sendl;

    }
    res.head( sys.m ) = free;
    
    if( sys.n ) {
        vec tmp = rhs.tail( sys.n ) - pimpl->PHinvPJT.transpose() * rhs.head( sys.m );

        // lambdas
        res.tail( sys.n ) =  pimpl->solver.solve( tmp );

        // constraint forces
        res.head( sys.m ) += pimpl->PHinvPJT * res.tail( sys.n);
        
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

