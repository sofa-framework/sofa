#ifndef COMPLIANT_UTILS_KKT_H
#define COMPLIANT_UTILS_KKT_H

#include <Eigen/Core>
#include <Eigen/Sparse>

namespace sofa {
namespace component {
namespace linearsolver {
struct AssembledSystem;
}
}
}

struct kkt {
				
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
		
		matrixC(const sys_type& sys) 
			: sys(sys) {
			
		}
		
		mutable vec result;
		
		template<class Vec>
		const vec& operator()(const Vec& x) const {
			result.noalias() = sys.C.selfadjointView<Eigen::Upper>() * x;
			return result;
		}
		
	};


private:
	matrixQ Q;
	matrixA A;
	matrixAT AT;
	matrixC C;

	unsigned m, n;
	
	bool parallel;
	
	mutable vec storage;
	
public:
	
	kkt( const sys_type& sys, bool parallel = false )
		: Q(sys),
		  A(sys),
		  AT(sys),
		  C(sys),
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
		}
		
		return storage;
	}

	template<class Vec> 
	const vec& omp_call(const Vec& x) const {
		
		if( n ) {
#pragma omp parallel sections
			{
#pragma omp section
				Q( x.head(m) );
#pragma omp section
				AT( x.tail(n) );
#pragma omp section
				A( x.head(m) );
#pragma omp section
				C( x.tail(n) );
			}
			
#pragma omp parallel sections
			{
#pragma omp section
				storage.head(m) = Q.result - AT.result;
#pragma omp section
				storage.tail(n) = -A.result - C.result;
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
