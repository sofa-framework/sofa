#include "LDLTSolver.h"

#include <sofa/core/ObjectFactory.h>

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <Eigen/LU>
#include <Eigen/Cholesky>
#include <Eigen/SVD>
using std::cerr;
using std::endl;

namespace sofa {
namespace component {
namespace linearsolver {

SOFA_DECL_CLASS(LDLTSolver);
int LDLTSolverClass = core::RegisterObject("Direct LDLT solver").add< LDLTSolver >();

typedef AssembledSystem::vec vec;






LDLTSolver::LDLTSolver() 
//    : damping( initData(&damping, (SReal)0.0, "damping", "damping lol") )
    : regularize( initData(&regularize, std::numeric_limits<real>::epsilon(), "regularize", "add identity*regularize to matrix H to make it definite."))
    , pimpl()
{
	
}

LDLTSolver::~LDLTSolver() {
	
}


void LDLTSolver::factor(const AssembledSystem& sys) {
	
    if( regularize.getValue() != (SReal)0.0 )
    {
        mat identity(sys.m,sys.m);
        identity.setIdentity();
        pimpl->Hinv.compute( sys.H + identity * regularize.getValue() );
    }
    else
        pimpl->Hinv.compute( sys.H );
	
	if( pimpl->Hinv.info() == Eigen::NumericalIssue ) {
        std::cerr << "LDLTSolver::factor: H is not psd. System solution will be wrong." << std::endl;
		
		std::cerr << pimpl->H << std::endl;
	}
	
	pimpl->dt = sys.dt;
	pimpl->m = sys.m;
	pimpl->n = sys.n;

	if( sys.n ) {
		pimpl_type::cmat schur(sys.n, sys.n);
//        pimpl_type::cmat PJT = sys.P.transpose() * sys.J.transpose();
        const pimpl_type::cmat& PJT = sys.J.transpose(); // H is already multiplied by P

		pimpl->HinvPJT.resize(sys.m, sys.n);
		pimpl->HinvPJT = pimpl->Hinv.solve( PJT );

		schur = (sys.C.transpose() + (PJT.transpose() * pimpl->HinvPJT )).selfadjointView<Eigen::Upper>();
		
		pimpl->schur.compute( schur );
		
		if( pimpl->schur.info() == Eigen::NumericalIssue ) {
            std::cerr << "LDLTSolver::factor: schur is not psd. System solution will be wrong." << std::endl;
			std::cerr << schur << std::endl;
		}
	} else {
		// nothing lol
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
        cerr<<"LDLTSolver::solve, Pv = " << Pv.transpose() << endl;
        cerr<<"LDLTSolver::solve, H = " << endl << dmat(sys.H) << endl;
    }

	// in place solve
	Pv = pimpl->Hinv.solve( Pv );
    if( debug.getValue() ){
        cerr<<"LDLTSolver::solve, solution = " << Pv.transpose() << endl;
        cerr<<"LDLTSolver::solve, verification = " << (sys.H * Pv).transpose() << endl;
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

