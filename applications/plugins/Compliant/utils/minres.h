#ifndef COMPLIANT_UTILS_MINRES_H
#define COMPLIANT_UTILS_MINRES_H

#include <cassert>
#include "krylov.h"

// License: LGPL 2.1
// Author: Maxime Tournier

// Adapted from Choi 2006, ITERATIVE METHODS FOR SINGULAR LINEAR
// EQUATIONS AND LEAST-SQUARES PROBLEMS

// example use:

// typedef minres<double> solver;
// solver::params p;
// p.iterations = 100;
// p.precision = 1e-16;

// solver::vec x, b;
// solver::solve(x, A, b);

template<class U>
struct minres  {
  
	typedef ::krylov<U> krylov;

	typedef typename krylov::vec vec;
	typedef typename krylov::real real;
	typedef typename krylov::params params;

	// solves (A - sigma I) x = b using minres. 
	// @A is a function object vec -> vec implementing matrix multiplication
	template<class Matrix>
	static void solve(vec& x, const Matrix& A, const vec& b, params& p, real sigma = 0) {
		
		vec residual = b;
    
		// deal with warm start
		if( !x.size() ) { 
			x = vec::Zero( b.size() );
		} else { 
			assert(x.size() == b.size() );
			residual -= A(x);
		}
    
		// easy peasy
		data d;
		d.residual( residual );

		unsigned i;
		for( i = 0; i < p.iterations; ++i) {
			if( d.phi <= p.precision) break;

            if(p.restart && i && i % p.restart == 0 ) {
                residual = b - A(x);
                d.residual( residual );
            }
            
			d.step(x, A, sigma);
		}

		// update outside world
		p.iterations = i;
		p.precision = d.phi;
    
	}
  

	// contains all the data needed for minres iterations
	struct data {
    
		unsigned n;			// dimension 
    
		real beta;

		vec v_prev, v, v_next;
      
		vec d, d_prev;
      
		real phi;			// residual norm
		real tau;
      
		real delta_1;
		real gamma_min; 		// minimum non-zero eigenvalue

		real norm;			// matrix norm
		real cond;			// condition number

		real c, s;

		real eps;

		unsigned k;			// iteration 
    
		// initializes minres given initial residual @r
		void residual(const vec& r) {
			n = r.size();
      
			const typename vec::ConstantReturnType zero = vec::Zero(n);
	
			beta = r.norm();
      
			// no residual, early exit
			if( !beta ) { 
				phi = 0;
				return;
			}
      
			v_prev = zero;
			v = r / beta;
			v_next = zero;

			d = zero;
			d_prev = zero;

			phi = beta;
			tau = beta;
	
			delta_1 = 0;
			gamma_min = 0;

			norm = 0;
			cond = 0;

			c = -1;
			s = 0;

			eps = 0;
			k = 1;
		}
    
		static real sign(const real& x) { return x < 0 ? -1 : 1; }
		static real abs(const real& x) { return x < 0 ? -x : x; }
    
		// lanczos iteration
		struct lanczos {
			real& alpha;
			real& beta;
			vec& v;

			lanczos(real& alpha, real& beta, vec& v)
				: alpha(alpha),
				  beta(beta),
				  v(v) {

			}

			// performs one lanczos step for (A - sigma I)x = b
			// @A is a function object vec -> vec
			template<class Matrix>
			static void step(lanczos res,
			                 const Matrix& A, 
			                 const vec& v,
			                 const vec& v_prev,
			                 real beta,
			                 real sigma)  {

				// use res.v as work vector
				vec& p = res.v;
	
				p = A(v) - sigma * v;
      
				res.alpha = v.dot( p );
      
				p -= res.alpha * v + beta * v_prev;

                // paranoid orthogonalization
                p -= p.dot(v) * v;
                
				res.beta = res.v.norm();
      
				if( res.beta ) res.v /= res.beta;
			}

		};

   
		// performs a small QR step ? TODO check this
		static void sym_ortho(const real& a, const real& b,
		                      real& c, real& s, real& r) {
      
			if( !b ) {
				s = 0;
				r = abs( a );
	
				c = a ? sign(a) : 1.0;
	
			} else if ( !a ) {
				c = 0;
				s = sign(b);
				r = abs(b);
			} else {

				const real aabs = abs(a);
				const real babs = abs(b);
      
				if( babs >= aabs ) {
					const real tau = a / b;

					s = sign( b ) / std::sqrt( 1 + tau * tau );
					c = s * tau;
					r = b / s;
				} else {
					// TODO should be  // else if( aabs > babs )
					const real tau = b / a;
	  
					c = sign( a ) / std::sqrt( 1 + tau * tau );
					s = c * tau;
					r = a / c;
				} 
			}
		}
    
    
		// performs one minres step for solving (A - sigma * I) x = b
		// @A is a function object vec -> vec
		template<class Matrix>
		void step(vec& x, const Matrix& A, real sigma = 0) {

			// solution already found lol !
			if( !phi ) return;
      
			real alpha;
			real beta_prev = beta;
      
			lanczos res(alpha, beta, v_next);
			lanczos::step(res, A, v, v_prev, beta, sigma);
      
			real delta_2 = c * delta_1  +  s * alpha;
			real gamma_1 = s * delta_1  -  c * alpha;

			real eps_next = s * beta;
			real delta_1_next = -c * beta;
	  
			real gamma_2;
			sym_ortho(gamma_1, beta, c, s, gamma_2);
      
			tau = c * phi;
			phi = s * phi;
	  
			norm = (k == 1) ? std::sqrt( alpha * alpha  +  beta * beta ) 
				: std::max(norm, std::sqrt( alpha * alpha  +  beta * beta  +  beta_prev * beta_prev));
	  
			if( gamma_2 ) {
				vec& d_next = v_prev;	// we use vprev as a temporary
	
				d_next = (v  -  delta_2 * d  -  eps * d_prev ) / gamma_2;
				x += tau * d_next;
	
				gamma_min = (k == 1)? gamma_2 : std::min(gamma_min, gamma_2);
				assert( gamma_min );
	  
				cond = norm / gamma_min;
	
				// pointer swaps instead of copies
				d_prev.swap( d );
				d.swap( d_next );
	
				v_prev.swap( v );
				v.swap( v_next );
	  
				eps = eps_next;
				delta_1 = delta_1_next;
	
			} else {
				// derp
			}
	   
			++k;
		}


	};

 
  


};



#endif
