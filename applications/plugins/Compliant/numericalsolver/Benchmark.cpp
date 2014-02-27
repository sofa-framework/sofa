#include "Benchmark.h"
#include <sofa/core/ObjectFactory.h>

#include "../assembly/AssembledSystem.h"
#include "Response.h"

#include "../utils/edit.h"


namespace sofa {
namespace component {
namespace linearsolver {

SOFA_DECL_CLASS(Benchmark);
int BenchmarkClass = core::RegisterObject("A benchmark for iterative solvers.").add< Benchmark >();

Benchmark::Benchmark()
	: factor(initData(&factor, "factor", "time elapsed during factor")),
	  solve(initData(&solve, "solve", "time elapsed during solve")),

	  primal(initData(&primal, "primal", "primal error")),	  
	  dual(initData(&dual, "dual", "dual error")),
	  complementarity(initData(&complementarity, "complementarity", "complementarity error"))
{
	
}


void Benchmark::clear() {
	
	edit(primal)->clear();
	edit(dual)->clear();
	edit(complementarity)->clear();
	
}

void Benchmark::push(const vec& primal,
					 const vec& dual) {
	assert( primal.size() == dual.size() );
	unsigned n = primal.size();

	real err_primal = primal.cwiseMin( vec::Zero(n) ).norm() / n;
	real err_dual = dual.cwiseMin( vec::Zero(n) ).norm() / n;
	real err_compl = std::abs(primal.dot(dual)) / n;
	
	edit(this->primal)->push_back(err_primal);
	edit(this->dual)->push_back(err_dual);
	edit(this->complementarity)->push_back(err_compl);

}

	
void Benchmark::lcp(const AssembledSystem& system, 
					const vec& rhs,
					const Response& response, 
					const vec& dual) {

	vec v(system.m);
	response.solve( v, system.J.transpose() * dual );
	
	push( system.J * v - rhs, dual );
}  


void Benchmark::debug() const {
	for(unsigned i = 0, n = primal.getValue().size(); i < n; ++i) {
		std::cerr << primal.getValue()[i] << " ";
	}
	std::cerr << std::endl;
}

}
}
}
