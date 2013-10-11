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
	const mat JP;
	schur(const sys_type& sys, const MatrixMinv& Minv) 
		: sys(sys),
		  Minv(Minv),
		  JP(sys.J * sys.P)
		{
		assert( sys.n );
	};

	mutable vec result, tmp1, tmp2;
	
	template<class Vec>
	const vec& operator()(const Vec& x) const {
		
		tmp2.noalias() = JP.transpose() * x;
		Minv.solve(tmp1, tmp2);
		tmp2.noalias() = sys.C * x;
		tmp1 += tmp2;
		result.noalias() = JP * tmp1;
		
		return result;
	};
	

};


#endif
