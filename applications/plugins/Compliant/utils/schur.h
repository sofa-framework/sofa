#ifndef SCHUR_H
#define SCHUR_H

#include <Eigen/Core>
#include <Eigen/Sparse>

namespace sofa {
namespace component {
namespace linearsolver {
class AssembledSystem;
}
}
}

template<class MatrixMinv>
struct schur {
	
	typedef sofa::component::linearsolver::AssembledSystem sys_type;
	
	typedef sys_type::real real;
    typedef sys_type::rmat rmat;
	typedef sys_type::vec vec;
	

	typedef MatrixMinv Minv_type;
	
	const sys_type& sys;
	const Minv_type& Minv;
    const rmat JP;
	const real damping;
	
	const vec* prec;

	schur(const sys_type& sys, 
		  const MatrixMinv& Minv,
		  real damping = 0) 
		: sys(sys),
		  Minv(Minv),
		  JP(sys.J * sys.P),
		  damping(damping),
		  prec(0)
		{
		assert( sys.n );
    }

	mutable vec result, tmp1, tmp2, tmp3;
	
	template<class Vec>
	const vec& operator()(const Vec& x) const {

		if( prec ) {
			tmp3 = prec->array() * x.array();
			tmp2.noalias() = JP.transpose() * tmp3;
		} else {
			tmp2.noalias() = JP.transpose() * x;
		}
		Minv.solve(tmp1, tmp2);
		result.noalias() = JP * tmp1;

		if( prec ) {
            result.noalias() += sys.C * tmp3;
			result = prec->array() * result.array();
		} else {
            result.noalias() += sys.C * x;
		}		
		
		if( damping ) result += damping * x;
		return result;
    }
	

};


#endif
