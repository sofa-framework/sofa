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
	
	typedef sys_type::real real;
	typedef sys_type::mat mat;
	typedef sys_type::vec vec;
	
	const sys_type& sys;
	const MatrixMinv& Minv;
	const mat JP;
	const real damping;
	
	schur(const sys_type& sys, 
		  const MatrixMinv& Minv,
		  real damping = 0) 
		: sys(sys),
		  Minv(Minv),
		  JP(sys.J * sys.P),
		  damping(damping)
		{
		assert( sys.n );
	};

	mutable vec result, tmp1, tmp2;
	
	template<class Vec>
	const vec& operator()(const Vec& x) const {
		
		tmp2.noalias() = JP.transpose() * x;
		Minv.solve(tmp1, tmp2);
		result.noalias() = JP * tmp1;
		result.noalias() = result + sys.C * x;
		
		if( damping ) result += damping * x;
		return result;
	};
	

};


#endif
