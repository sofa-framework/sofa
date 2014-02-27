#ifndef COMPLIANT_MISC_BENCHMARK_H
#define COMPLIANT_MISC_BENCHMARK_H

#include "../initCompliant.h"
#include <sofa/core/objectmodel/BaseObject.h>

#include <Eigen/Core>

namespace sofa {
namespace component {
namespace linearsolver {

class AssembledSystem;
class Response;

class SOFA_Compliant_API Benchmark : public core::objectmodel::BaseObject {

  public:

	// TODO maybe add a 'size' data to hold vector size. 'clear'
	// should not dealloc/realloc, so we should be fine.
	
	// duration
	Data<SReal> factor, solve;
	
	// convergence
	Data< vector<SReal> > dual, primal, complementarity;
	
	typedef SReal real;
	typedef Eigen::Matrix<real, Eigen::Dynamic, 1> vec;

    SOFA_CLASS(Benchmark, core::objectmodel::BaseObject);

	Benchmark();

	void clear();
	void debug() const;


	// push the results for the last iteration, for an LCP solver
	void lcp(const AssembledSystem& system, 
			 const vec& rhs,	// the lcp rhs: b - J Minv f
			 const Response& response, 
			 const vec& dual); 
	
	// push the results for the last iteration, for a QP solver
	void qp(const AssembledSystem& system, 
			const vec& rhs,
			const vec& x);

  protected:

	void push(const vec& primal, const vec& dual);
	
};



}
}
}


#endif
