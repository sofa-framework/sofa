#ifndef SCHUR_H
#define SCHUR_H

#include <Eigen/Core>
#include <Eigen/Sparse>

namespace sofa {
namespace component {
namespace linearsolver {
struct AssembledSystem;
}
}
}

template<class MatrixMinv>
struct schur {
	
	typedef sofa::component::linearsolver::AssembledSystem sys_type;
	
	typedef sys_type::mat mat;
	typedef sys_type::vec vec;
	
	const sys_type& sys;
	const MatrixMinv& Minv;

	schur(const sys_type& sys, const MatrixMinv& Minv) 
		: sys(sys),
		  Minv(Minv) {
		assert( sys.n );
	};

	mutable vec result, tmp1, tmp2;
	
	const vec& operator()(const vec& x) const {

		tmp1.noalias() = sys.J.transpose() * x;
		tmp2.noalias() = sys.P * tmp1;
		tmp1.noalias() = Minv.solve(tmp2);
		tmp2.noalias() = sys.P * tmp1;
		result.noalias() = sys.J * tmp2;
		
		return result;
	};
	

};


#endif
