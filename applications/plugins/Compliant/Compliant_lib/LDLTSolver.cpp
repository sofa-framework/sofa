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
	: pimpl(),
	  damping( initData(&damping,
	                    0.0,
	                    "damping",
	                    "damping lol") ) 
{
	
}

LDLTSolver::~LDLTSolver() {
	
}


// right-shift, size x (total_cols) matrix: (0, id, 0)
static AssembledSystem::mat shift_right(unsigned off, unsigned size, unsigned total_cols) {
	AssembledSystem::mat res( size, total_cols); 
	assert( total_cols >= (off + size) );
	
	// res.resize(size, total_cols );
	// res.reserve(Eigen::VectorXi::Constant(size, 1));
	res.reserve( size );
	
	for(unsigned i = 0; i < size; ++i) {
		res.startVec( i );
		res.insertBack(i, off + i) = 1.0;
		// res.insert(i, off + i) = 1.0;
	}
	res.finalize();
	
	res.makeCompressed(); // TODO is this needed ?
	return res;
}


void LDLTSolver::factor(const AssembledSystem& sys) {
	
    pimpl->H = sys.H.selfadjointView<Eigen::Upper>();

	pimpl->Hinv.compute( pimpl->H );
	
	if( pimpl->Hinv.info() == Eigen::NumericalIssue ) {
		std::cerr << "H is not psd :-/" << std::endl;

		std::cerr << pimpl->H << std::endl;
	}
	
	pimpl->dt = sys.dt;
	pimpl->m = sys.m;
	pimpl->n = sys.n;

	if( sys.n ) {
		pimpl_type::cmat schur(sys.n, sys.n);
		pimpl_type::cmat JT = sys.J.transpose(); 
		
		pimpl->HinvJT.resize(sys.m, sys.n);
		pimpl->HinvJT = pimpl->Hinv.solve( JT );
		
		schur = (sys.C.transpose() + (sys.J * pimpl->HinvJT )).selfadjointView<Eigen::Upper>();
		
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
                         const AssembledSystem& system,
                         const AssembledSystem::vec& rhs) const {
	assert( res.size() == pimpl->m + pimpl->n);
	assert( rhs.size() == pimpl->m + pimpl->n);

	unsigned m = pimpl->m;
	unsigned n = pimpl->n;
	

	SReal alpha = 1 / ( 1 + damping.getValue() * pimpl->dt );
	
	res.head( m ) = pimpl->Hinv.solve( alpha * rhs.head(m) );

	// std::cerr << "LDLTSolver, rhs" << std::endl
	//           << rhs.transpose() << std::endl;

	if( n ) {
		// lambdas
		res.tail( n ) = pimpl->schur.solve( pimpl->HinvJT.transpose() * rhs.head( m ) - rhs.tail( n ) );
		
		// constraint forces
		res.head( m ) -= pimpl->HinvJT * res.tail(n);
	} 
	
} 


}
}
}

