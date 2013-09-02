#ifndef MINRESSOLVER_INL
#define MINRESSOLVER_INL

#include "KrylovSolver.h"


#include "utils/scoped.h"

namespace sofa {
namespace component {
namespace linearsolver {


template<class NumericalSolver>
KrylovSolver<NumericalSolver>::KrylovSolver()
    : KKTSolver(),
      precision(initData(&precision,
	                     SReal(1e-3),
	                     "precision",
	                     "residual norm threshold")),
	  iterations(initData(&iterations,
	                      unsigned(10),
	                      "iterations",
	                      "iteration bound")),
	  relative(initData(&relative, true, "relative", "use relative precision") ),
	  verbose(initData(&verbose, false, "verbose", "print debug stuff on std::cerr") )
{
	
}
			
namespace {

struct kkt {
				
	typedef AssembledSystem sys_type;
					
	const sys_type& sys;
	
	kkt(const sys_type& sys) 
	: sys(sys) {
		
		storage.resize( sys.m + sys.n );

		proj.resize( sys.m );
		tmp.resize( sys.m );
		
	}

	typedef sys_type::vec vec;
	mutable vec storage, proj, tmp;
				
	// matrix-vector product
	const vec& operator()(const vec& x) const {
		proj.noalias() = sys.P.selfadjointView<Eigen::Upper>() * x.head( sys.m );
		tmp.noalias() = sys.H.selfadjointView<Eigen::Upper>() * proj;
		
		storage.head( sys.m ).noalias() =  sys.P.selfadjointView<Eigen::Upper>() * tmp;
		
		if( sys.n ) {
			tmp.noalias() = sys.J.transpose() * x.tail( sys.n );
			storage.head( sys.m ).noalias() = storage.head( sys.m )  +  sys.P * tmp;
			
			// TODO are there temporaries allocated here ?
			storage.tail( sys.n ).noalias() = sys.J * proj  -  sys.C.selfadjointView<Eigen::Upper>() * x.tail( sys.n );
		}
		
		return storage;
	}

};
}
			
template<class NumericalSolver>
void KrylovSolver<NumericalSolver>::factor(const AssembledSystem& ) { }

template<class NumericalSolver>
void KrylovSolver<NumericalSolver>::solve(AssembledSystem::vec& x,
                      const AssembledSystem& system,
                      const AssembledSystem::vec& b) const {
	// TODO timer is not thread safe :-/
	// scoped::timer step("system solve");

    typename NumericalSolver::params p;
	p.iterations = iterations.getValue();
	p.precision = precision.getValue();
				
	if( relative.getValue() ) p.precision *= b.norm();
				
	kkt A(system);
    NumericalSolver::solve(x, A, b, p);
	
	if( verbose.getValue() ) {
		std::cerr << "minres: " << p.iterations << " iterations, absolute residual: " << (A(x) - b).norm() << std::endl;
		
	}
}

			
}
}
}


#endif // MINRESSOLVER_INL
