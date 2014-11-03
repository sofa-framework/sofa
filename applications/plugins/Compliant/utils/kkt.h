#ifndef COMPLIANT_UTILS_KKT_H
#define COMPLIANT_UTILS_KKT_H

#include <Eigen/Core>
#include <Eigen/Sparse>


namespace sofa {
namespace component {
namespace linearsolver {
class AssembledSystem;
}
}
}

struct kkt {
				
    // Should not these types be templates? to let utils/ktt.h outside of sofa
    typedef sofa::component::linearsolver::AssembledSystem sys_type;

	typedef sys_type::real real;
	typedef sys_type::mat mat;
	typedef sys_type::vec vec;

	struct matrixQ {
		
		const sys_type& sys;
		
		matrixQ(const sys_type& sys) 
			: sys(sys) { 

		}
		
		mutable vec tmp, result;
		
		template<class Vec>
		const vec& operator()(const Vec& x) const {
			result.noalias() = sys.P.selfadjointView<Eigen::Upper>() * x;
			tmp.noalias() = sys.H.selfadjointView<Eigen::Upper>() * result;
			result.noalias() = sys.P.selfadjointView<Eigen::Upper>() * tmp;
			return result;
		}
	};


	struct matrixA {
		const sys_type& sys;

		matrixA(const sys_type& sys) 
			: sys(sys) { 

		}
		
		mutable vec tmp, result;
		
		template<class Vec>
		const vec& operator()(const Vec& x) const {
			tmp.noalias() = sys.P.selfadjointView<Eigen::Upper>() * x;
			result.noalias() = sys.J * tmp;
			return result;
		}
		
	};


	struct matrixAT {
		const sys_type& sys;
		
		matrixAT(const sys_type& sys) 
			: sys(sys) { 
			
		}
		
		mutable vec tmp, result;
		
		template<class Vec>
		const vec& operator()(const Vec& x) const {
			tmp.noalias() = sys.J.transpose() * x;
			result.noalias() = sys.P * tmp;
			return result;
		}
		
	};
	

	struct matrixC {
		const sys_type& sys;
		const real damping;
		
		matrixC(const sys_type& sys,
				real damping = 0) 
			: sys(sys),
			  damping(damping) {
			
		}
		
		mutable vec result;
		
		template<class Vec>
		const vec& operator()(const Vec& x) const {
			result.noalias() = sys.C.selfadjointView<Eigen::Upper>() * x;
			if( damping ) result += damping * x;
			return result;
		}
		
	};




private:
	matrixQ Q;
	matrixA A;
	matrixAT AT;
    matrixC C;

	unsigned m, n;
	

	mutable vec storage;
	
public:
	bool parallel;
	
    kkt( const sys_type& sys, bool parallel = false, real damping = 0)
		: Q(sys),
		  A(sys),
		  AT(sys),
          C(sys, damping),
		  m( sys.m ),
          n( sys.n ),
		  parallel(parallel) {

		storage.resize( m + n );
		
	}


private:
	template<class Vec>
	const vec& call(const Vec& x) const {
		
		storage.head(m) = Q(x.head(m));

		if( n ) {
			storage.head(m) -= AT( x.tail(n) );
			storage.tail(n) = - A( x.head(m) ) - C(x.tail(n));
			// if( damping ) storage.tail(n) -= damping * x.tail(n);
		}
		
		return storage;
	}

	template<class Vec> 
	const vec& omp_call(const Vec& x) const {
		
		if( n ) {
#ifdef USING_OMP_PRAGMAS
#pragma omp parallel sections
#endif
			{
#ifdef USING_OMP_PRAGMAS
#pragma omp section
#endif
				Q( x.head(m) );
#ifdef USING_OMP_PRAGMAS
#pragma omp section
#endif
				AT( x.tail(n) );
#ifdef USING_OMP_PRAGMAS
#pragma omp section
#endif
				A( x.head(m) );
#ifdef USING_OMP_PRAGMAS
#pragma omp section
#endif
				C( x.tail(n) );
			}

#ifdef USING_OMP_PRAGMAS
#pragma omp parallel sections
#endif
			{
#ifdef USING_OMP_PRAGMAS
#pragma omp section
#endif
				storage.head(m) = Q.result - AT.result;
#ifdef USING_OMP_PRAGMAS
#pragma omp section
#endif
				storage.tail(n) = -A.result - C.result;

				// if( damping ) storage.tail(n) -= damping * x.tail(n);
			}
			
			} else {
				storage.head(m) = Q(x.head(m));
			}
		
				return storage;
	 }

public:
	template<class Vec>
	const vec& operator()(const Vec& x) const {
		return parallel ? omp_call(x) : call(x);
	}
	
};

struct kkt_opt {
	typedef sofa::component::linearsolver::AssembledSystem sys_type;
	
	typedef sys_type::mat mat;
	typedef sys_type::vec vec;
	
	const sys_type& sys;
	mutable vec result, tmp, Pv, HPv, JTlambda, JPv, Clambda ;
	
	const unsigned m;
	const unsigned n;
	
	kkt_opt(const sys_type& sys) 
		: sys(sys),
		  m(sys.m),
		  n(sys.n)
		{ 
		result.resize( m + n );
		assert( n );
		// std::cerr << sys.C << std::endl; 
	}
	

	template<class Vec>
	const vec& operator()(const Vec& x) const {
		Pv.noalias() = sys.P * x.head(m);
		
		// parallelizable 
		{
			HPv.noalias() = sys.H * Pv;
			
			JTlambda.noalias() = sys.J.transpose() * x.tail(n);
			JPv.noalias() = sys.J * Pv;
			Clambda.noalias() = sys.C * x.tail(n);
		}
		
		tmp.noalias() = HPv - JTlambda;
		
		result.head(sys.m).noalias() = sys.P * tmp;
		result.tail(sys.n).noalias() = -JPv - Clambda;
		
		return result;
	}
};




#endif
