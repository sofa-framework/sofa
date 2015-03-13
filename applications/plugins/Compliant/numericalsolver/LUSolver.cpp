#include "LUSolver.h"
#include "LUResponse.h"

#include <sofa/core/ObjectFactory.h>


#include "../utils/scoped.h"

using std::cerr;
using std::endl;

namespace sofa {
namespace component {
namespace linearsolver {

SOFA_DECL_CLASS(LUSolver)
static int LUSolverClass = core::RegisterObject("Direct LU solver").add< LUSolver >();

typedef AssembledSystem::vec vec;



LUSolver::LUSolver()
{

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
        std::cout << "LUSolver: fallback response class: "
                  << response->getClassName()
                  << " added to the scene" << std::endl;
    }
}



void LUSolver::factor_schur( const pimpl_type::cmat& schur )
{
    if( debug.getValue() ){
        typedef AssembledSystem::dmat dmat;
        cerr<< "LUSolver::factor, HinvPJT = " << endl << dmat(pimpl->HinvPJT) << endl;
        cerr<< "LUSolver::factor, schur = " << endl << dmat(schur) << endl;
    }

    {
        scoped::timer step("Schur factorization");
        pimpl->schur.compute( schur );
    }

    if( pimpl->schur.info() == Eigen::NumericalIssue ){
        std::cerr << "LUSolver::factor: schur factorization failed." << std::endl;
        std::cerr << schur << std::endl;
    }
}

void LUSolver::factor(const AssembledSystem& sys) {

    // response matrix
    assert( response );

    if( !sys.isPIdentity ) // there are projective constraints
    {
        response->factor( sys.P.transpose() * sys.H * sys.P, true ); // replace H with P^T.H.P to account for projective constraints

        if( sys.n ) // bilateral constraints
        {
            pimpl_type::cmat PJT = sys.P.transpose() * sys.J.transpose(); //yes, we have to filter J, even if H is filtered already. Otherwise Hinv*JT has large (if not infinite) values on filtered DOFs
            response->solve( pimpl->HinvPJT, PJT );
            factor_schur(sys.C.transpose() + (PJT.transpose() * pimpl->HinvPJT ));
        }
    }
    else // no projection
    {
        response->factor( sys.H );

        if( sys.n ) // bilateral constraints
        {
            response->solve( pimpl->HinvPJT, sys.J.transpose() );
            factor_schur( sys.C.transpose() + ( sys.J * pimpl->HinvPJT ) );
        }
    }

}


void LUSolver::solve(AssembledSystem::vec& res,
                       const AssembledSystem& sys,
                       const AssembledSystem::vec& rhs) const {

    assert( res.size() == sys.size() );
    assert( rhs.size() == sys.size() );


    vec Pv = (sys.P * rhs.head(sys.m));

    typedef AssembledSystem::dmat dmat;

    if( debug.getValue() ){
        cerr<<"LUSolver::solve, rhs = " << rhs.transpose() << endl;
        cerr<<"LUSolver::solve, Pv = " << Pv.transpose() << endl;
        cerr<<"LUSolver::solve, H = " << endl << dmat(sys.H) << endl;
    }

    // in place solve
    response->solve( Pv, Pv );

    if( debug.getValue() ){
        cerr<<"LUSolver::solve, free motion solution = " << Pv.transpose() << endl;
        cerr<<"LUSolver::solve, verification = " << (sys.H * Pv).transpose() << endl;
        cerr<<"LUSolver::solve, sys.m = " << sys.m << ", sys.n = " << sys.n << ", rhs.size = " << rhs.size() << endl;

    }
    res.head( sys.m ) = sys.P * Pv;

    if( sys.n ) {
        vec tmp = rhs.tail( sys.n ) - pimpl->HinvPJT.transpose() * rhs.head( sys.m );


        // lambdas
        res.tail( sys.n ) = pimpl->schur.solve( tmp );

        // constraint forces
        res.head( sys.m ) += sys.P * (pimpl->HinvPJT * res.tail( sys.n));
        if( debug.getValue() ){
            cerr<<"LUSolver::solve, free motion constraint error= " << -tmp.transpose() << endl;
            cerr<<"LUSolver::solve, lambda = " << res.tail(sys.n).transpose() << endl;
            cerr<<"LUSolver::solve, constraint forces = " << (sys.P * (pimpl->HinvPJT * res.tail( sys.n))).transpose() << endl;
        }
    }

} 


}
}
}

