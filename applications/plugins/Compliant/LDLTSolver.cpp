#include "LDLTSolver.h"

#include <sofa/core/ObjectFactory.h>

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <Eigen/LU>
#include <Eigen/Cholesky>
#include <Eigen/SVD>

namespace sofa {
namespace component {
namespace linearsolver {

SOFA_DECL_CLASS(LDLTSolver);
int LDLTSolverClass = core::RegisterObject("Direct LDLT solver").add< LDLTSolver >();

typedef AssembledSystem::vec vec;






LDLTSolver::LDLTSolver() 
    : damping( initData(&damping, 0.0, "damping", "damping lol") )
    , pimpl()
{
	
}

LDLTSolver::~LDLTSolver() {
	
}


void LDLTSolver::factor(const AssembledSystem& sys) {
	
	pimpl->Hinv.compute( sys.H );
	
	if( pimpl->Hinv.info() == Eigen::NumericalIssue ) {
		std::cerr << "H is not psd :-/" << std::endl;
		
		std::cerr << pimpl->H << std::endl;
	}
	
	pimpl->dt = sys.dt;
	pimpl->m = sys.m;
	pimpl->n = sys.n;

	if( sys.n ) {
		pimpl_type::cmat schur(sys.n, sys.n);
		pimpl_type::cmat PJT = sys.P.transpose() * sys.J.transpose(); 
		
		pimpl->HinvPJT.resize(sys.m, sys.n);
		pimpl->HinvPJT = pimpl->Hinv.solve( PJT );

		schur = (sys.C.transpose() + (PJT.transpose() * pimpl->HinvPJT )).selfadjointView<Eigen::Upper>();
		
		pimpl->schur.compute( schur );
		
		if( pimpl->schur.info() == Eigen::NumericalIssue ) {
			std::cerr << "schur is not psd :-/" << std::endl;
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
	

	SReal alpha = 1.0 / ( 1.0 + damping.getValue() * pimpl->dt );
	
	vec tmp = alpha * (sys.P * rhs.head(sys.m));
	
	// in place solve
	tmp = pimpl->Hinv.solve(tmp);
	
	res.head( sys.m ) = sys.P * tmp;

	if( sys.n ) {
		vec tmp = rhs.tail( sys.n ) - sys.P * (pimpl->HinvPJT.transpose() * rhs.head( sys.m ));

		// lambdas
		res.tail( sys.n ) = pimpl->schur.solve( tmp );
		
		// constraint forces
		res.head( sys.m ) += sys.P * (pimpl->HinvPJT * res.tail( sys.n));
	} 
	
} 


}
}
}

