#include "Benchmark.h"
#include <sofa/core/ObjectFactory.h>

#include "../assembly/AssembledSystem.h"
#include "Response.h"

#include "../utils/edit.h"


namespace sofa {
namespace component {
namespace linearsolver {

SOFA_DECL_CLASS(Benchmark)
int BenchmarkClass = core::RegisterObject("A benchmark for iterative solvers.").add< Benchmark >();

Benchmark::Benchmark()
	: factor(initData(&factor, "factor", "time elapsed during factor")),

	  primal(initData(&primal, "primal", "primal error")),	  
	  dual(initData(&dual, "dual", "dual error")),
	  complementarity(initData(&complementarity, "complementarity", "complementarity error")),
	  optimality(initData(&optimality, "optimality", "optimality error")),
	  duration(initData(&duration, "duration", "cumulated solve time"))
{
	
}

void Benchmark::clear() {
	
    editOnly(primal)->clear();
    editOnly(dual)->clear();
    editOnly(complementarity)->clear();
    editOnly(optimality)->clear();
    editOnly(duration)->clear();
	
}

void Benchmark::push(const vec& primal,
					 const vec& dual) {
	assert( primal.size() == dual.size() );
	unsigned n = primal.size();

	real err_primal = primal.cwiseMin( vec::Zero(n) ).norm() / n;
	real err_dual = dual.cwiseMin( vec::Zero(n) ).norm() / n;
	real err_compl = std::abs(primal.dot(dual)) / n;
	
    editOnly(this->primal)->push_back(err_primal);
    editOnly(this->dual)->push_back(err_dual);
    editOnly(this->complementarity)->push_back(err_compl);

}


unsigned Benchmark::elapsed() const {
    using namespace std::chrono;
	clock_type::time_point now = clock_type::now();
	
	return duration_cast<microseconds> (now - last).count();
}



unsigned Benchmark::restart() {
    using namespace std::chrono;
	clock_type::time_point now = clock_type::now();
	
	unsigned res = ( duration_cast<microseconds> (now - last) ).count();

	last = now;
	return res;
}


	
void Benchmark::lcp(const AssembledSystem& system, 
					const vec& rhs,
					const Response& response, 
					const vec& dual,
					const vec* prec) {
	edit(this->duration)->push_back(elapsed());

	vec lambda = dual.head(system.n);
	if( prec ) lambda = prec->array() * lambda.array();

	vec v(system.m);
	response.solve( v, system.P * (system.J.transpose() * lambda) );
	
	push( system.J * (system.P * v) - rhs, lambda );
	edit(this->optimality)->push_back( 0 );
	restart();
}  


void Benchmark::qp(const AssembledSystem& system, 
				   const vec& rhs,
				   const vec& x) {
	edit(this->duration)->push_back(elapsed());
	
	push( system.J * system.P * x.head(system.m) - rhs.tail(system.n), x.tail(system.n) );
	
	vec opt = system.P * (system.H * (system.P * x.head(system.m)) 
						  - system.J.transpose() * x.tail(system.n) - rhs.head(system.m));

	edit(this->optimality)->push_back( opt.norm() / system.m );
	restart();
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
