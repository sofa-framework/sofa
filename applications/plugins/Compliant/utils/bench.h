#ifndef UTILS_BENCH_H
#define UTILS_BENCH_H

#include "../assembly/AssembledSystem.h"

namespace sofa {
namespace component {
namespace linearsolver {


struct bench {
	typedef AssembledSystem system_type;

	typedef system_type::real real;
	typedef system_type::vec vec;
	typedef system_type::mat mat;

	// std::vector< real > residual;

	
	// residual
	static real lcp(const vec& primal,
	                const vec& dual) {
		unsigned n = dual.size();
		
		real res = 
			// primal error
			primal.cwiseMin( vec::Zero(n) ).norm() +

			// dual error
			dual.cwiseMin( vec::Zero(n) ).norm() +

			// complementarity
			std::abs( primal.dot(dual) );
		
		// std::cerr << "lcp error: " << res << std::endl;

		return res;
	}


	struct object {
		const std::string filename;

		object(const std::string& filename) : filename(filename) { }

		~object() { 
			write( filename );
		}
		
		typedef std::vector<real> frame_type;
		mutable std::vector< frame_type > result;
		
		void write(const std::string& filename ) const {
			std::ofstream out(filename.c_str());

			for(unsigned i = 0, n = result.size(); i < n; ++i) {
				for(unsigned j = 0, m = result[i].size(); j < m; ++j) {
					out << result[i][j] << ' '; 
				}
				out << '\n';
			}
			
		}
	};

	
};


struct lcp_bench : bench::object {

	const AssembledSystem& system;
	const Response& response;

	typedef bench::vec vec;
	typedef bench::real real;

	const vec unconstrained;
	const vec b;

	lcp_bench(const std::string& filename, 
	          const AssembledSystem& system,
	          const Response& response,
	          const vec& unconstrained,
	          const vec& b)
		: bench::object(filename),
		  system(system),
		  response(response),
		  unconstrained(unconstrained),
		  b(b) {
	}
	
	real operator()(const vec& lambda) const {
		// std::cout << "called yo ! " << result.size() << " - " << lambda.transpose() << std::endl;

		vec tmp = lambda;
		
		vec v(system.m);
		response.solve( v, system.J.transpose() * tmp );
		v += unconstrained;
		
		result.push_back( frame_type() );
		frame_type& back = result.back();
		
		vec primal = system.J * v - b;
		vec dual = tmp;

		real err_primal = primal.cwiseMin( vec::Zero(system.n) ).norm() / system.n;
		real err_dual = dual.cwiseMin( vec::Zero(system.n) ).norm() / system.n;
		real err_compl = std::abs(primal.dot(dual)) / system.n;
		
		real total = err_primal + err_dual + err_compl;
		
		back.push_back( result.size() );
		back.push_back( total );
		back.push_back( err_primal );
		back.push_back( err_dual );
		back.push_back( err_compl );
		
		return total;
	}
};


struct qp_bench : bench::object {

	const AssembledSystem& system;
	
	typedef bench::vec vec;
	typedef bench::real real;

	const vec rhs;
	
	qp_bench(const std::string& filename, 
	         const AssembledSystem& system,
	         const vec& rhs)
		: bench::object(filename),
		  system(system),
		  rhs( rhs ) {		  
	}
	
	real operator()(const vec& x) const {

		assert( x.size() == system.size() );
		
		result.push_back( frame_type() );
		frame_type& back = result.back();
		back.push_back( result.size() );
		
		if( ! system.n ) {
			back.push_back( (system.H * x.head(system.m) - rhs.head(system.m)).norm() / system.m );
			return back.back();
		}
		
		vec tmp = x;

		vec primal = system.J * tmp.head(system.m) - rhs.tail(system.n);
		vec dual = tmp.tail(system.n);
		
		real err_primal = primal.cwiseMin( vec::Zero(system.n) ).norm() / system.n;
		real err_dual = dual.cwiseMin( vec::Zero(system.n) ).norm() / system.n;
		real err_compl = std::abs(primal.dot(dual)) / system.n;
		
		real err_opt = (system.H * tmp.head(system.m) - system.J.transpose() * dual - rhs.head(system.m)).norm() / system.m;
		
		real total = err_primal + err_dual + err_compl + err_opt;

		back.push_back( total );
		back.push_back( err_primal );
		back.push_back( err_dual );
		back.push_back( err_compl );
		back.push_back( err_opt );
		
		return total;
	}
};


}
}
}

#endif
