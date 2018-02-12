#ifndef COMPLIANT_MISC_BENCHMARK_H
#define COMPLIANT_MISC_BENCHMARK_H

#include <Compliant/config.h>
#include <sofa/core/objectmodel/BaseObject.h>

#include <Eigen/Core>

#include <chrono>

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
	Data<SReal> factor; ///< time elapsed during factor
	
	// convergence
    Data< helper::vector<SReal> > primal; ///< primal error
    Data< helper::vector<SReal> > dual; ///< dual error
    Data< helper::vector<SReal> > complementarity; ///< complementarity error
    Data< helper::vector<SReal> > optimality; ///< optimality error
    Data< helper::vector<SReal> > duration; ///< cumulated solve time
	
	typedef SReal real;
	typedef Eigen::Matrix<real, Eigen::Dynamic, 1> vec;

    SOFA_CLASS(Benchmark, core::objectmodel::BaseObject);

	Benchmark();

	// clear benchmark data. call this at the beginning of solve.
	void clear();
	
	// 
	void debug() const;

    // restart timer and return elapsed microseconds since last call
	unsigned restart();

	// microseconds elapsed since last restart call
	unsigned elapsed() const;

	// push the results for the last iteration, including elapsed
	// time, for an LCP solver. restart timer at the end.
	void lcp(const AssembledSystem& system, 
			 const vec& rhs,	// the lcp rhs: b - J Minv f
			 const Response& response, 
			 const vec& dual,
			 const vec* prec = 0); 
	
	// push the results for the last iteration, including elapsed
	// time, for a QP solver. restart timer at the end.
	void qp(const AssembledSystem& system, 
			const vec& rhs,
			const vec& x);

	// convenience scoped timing tool
	struct scoped_timer {
		Benchmark* bench;
		Data<SReal> Benchmark::* member;

		// write elapsed time to member pointer on destruction
		scoped_timer(Benchmark* bench,
					 Data<SReal> Benchmark::* member)
			: bench(bench),
			  member(member) {
			if( bench ) bench->restart();
		}

		~scoped_timer() {
			if( bench ) {

				(bench->*member).setValue( bench->restart() );
			}
		}

	};

  protected:
	
    typedef std::chrono::high_resolution_clock clock_type;
	clock_type::time_point last;

	void push(const vec& primal, const vec& dual);
	
};



}
}
}


#endif
