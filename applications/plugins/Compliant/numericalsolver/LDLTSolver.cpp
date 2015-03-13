#include "LDLTSolver.h"
#include "LDLTResponse.h"

#include <sofa/core/ObjectFactory.h>

#include "../utils/scoped.h"

using std::cerr;
using std::endl;

namespace sofa {
namespace component {
namespace linearsolver {

SOFA_DECL_CLASS(LDLTSolver)
int LDLTSolverClass = core::RegisterObject("Direct LDLT solver").add< LDLTSolver >();

typedef AssembledSystem::vec vec;






LDLTSolver::LDLTSolver() 
    : KKTSolver()
    , pimpl()
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
        std::cout << "LDLTSolver: fallback response class: "
                  << response->getClassName()
                  << " added to the scene" << std::endl;
    }
}



void LDLTSolver::factor_schur( const pimpl_type::cmat& schur )
{
    if( debug.getValue() ){
        typedef AssembledSystem::dmat dmat;
        cerr<< "LDLTSolver::factor, HinvPJT = " << endl << dmat(pimpl->HinvPJT) << endl;
        cerr<< "LDLTSolver::factor, schur = " << endl << dmat(schur) << endl;
    }

    {
        scoped::timer step("Schur factorization");
        pimpl->schur.compute( schur.selfadjointView<Eigen::Upper>() );
    }

    if( pimpl->schur.info() == Eigen::NumericalIssue ){
        std::cerr << "LDLTSolver::factor: schur is not psd. System solution will be wrong." << std::endl;
        std::cerr << schur << std::endl;
    }
}

void LDLTSolver::factor(const AssembledSystem& sys) {

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


void LDLTSolver::solve(AssembledSystem::vec& res,
                       const AssembledSystem& sys,
                       const AssembledSystem::vec& rhs) const {

    assert( res.size() == sys.size() );
    assert( rhs.size() == sys.size() );


    vec Pv = (sys.P * rhs.head(sys.m));

    typedef AssembledSystem::dmat dmat;

    if( debug.getValue() ){
        cerr<<"LDLTSolver::solve, rhs = " << rhs.transpose() << endl;
        cerr<<"LDLTSolver::solve, Pv = " << Pv.transpose() << endl;
        cerr<<"LDLTSolver::solve, H = " << endl << dmat(sys.H) << endl;
    }

    // in place solve
    response->solve( Pv, Pv );

    if( debug.getValue() ){
        cerr<<"LDLTSolver::solve, free motion solution = " << Pv.transpose() << endl;
        cerr<<"LDLTSolver::solve, verification = " << (sys.H * Pv).transpose() << endl;
        cerr<<"LDLTSolver::solve, sys.m = " << sys.m << ", sys.n = " << sys.n << ", rhs.size = " << rhs.size() << endl;

    }
    res.head( sys.m ) = sys.P * Pv;

    if( sys.n ) {
        vec tmp = rhs.tail( sys.n ) - pimpl->HinvPJT.transpose() * rhs.head( sys.m );


        // lambdas
        res.tail( sys.n ) = pimpl->schur.solve( tmp );

        // constraint forces
        res.head( sys.m ) += sys.P * (pimpl->HinvPJT * res.tail( sys.n));
        if( debug.getValue() ){
            cerr<<"LDLTSolver::solve, free motion constraint error= " << -tmp.transpose() << endl;
            cerr<<"LDLTSolver::solve, lambda = " << res.tail(sys.n).transpose() << endl;
            cerr<<"LDLTSolver::solve, constraint forces = " << (sys.P * (pimpl->HinvPJT * res.tail( sys.n))).transpose() << endl;
        }
    }

} 


}
}
}

