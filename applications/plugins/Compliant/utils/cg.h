#ifndef COMPLIANT_UTILS_CG_H
#define COMPLIANT_UTILS_CG_H

#include <cassert>
#include "krylov.h"

// License: LGPL 2.1
// Author: Maxime Tournier

// conjugate gradient solver

// example use:

// typedef cg<double> solver;
// solver::params p;
// p.iterations = 100;
// p.precision = 1e-16;

// solver::vec x, b;
// solver::solve(x, A, b);

template<class U>
struct cg
{

    typedef ::krylov<U> krylov;

    typedef typename krylov::vec vec;
    typedef typename krylov::real real;
    typedef typename krylov::natural natural;
    typedef typename krylov::params params;

    // solves Ax = b using cg.
    // @A is a function object vec -> vec implementing matrix multiplication
    template<class Matrix>
    static void solve(vec& x, const Matrix& A, const vec& b, params& p)
    {

        vec residual = b;

        // deal with warm start
        if( !x.size() )
        {
            x = vec::Zero( b.size() );
        }
        else
        {
            assert(x.size() == b.size() );
            residual -= A(x);
        }

        // easy peasy
        data d;

		d.residual( residual );

        natural i;
        for( i = 0; i < p.iterations && d.phi > p.precision; ++i)
        {
            d.step(x, A);
        }
        p.iterations = i;
        p.precision = d.phi;

    }


    // contains all the data needed for cg iterations
    struct data
    {

//        natural n;			// dimension

        vec p;			// descent direction
        vec r;			// residual
        vec Ap;			// A(p)

        real phi2;		// residual squared norm
        real phi;			// residual norm

        natural k;		// iteration

        void residual( const vec& rr ) {
			r = rr;
            p = r;
			
            phi2 = r.squaredNorm();
            phi = std::sqrt( phi2 );
			
            k = 1;
        }


        // performs one cg step. returns false on singularity, true
        // otherwise. @A is a function object vec -> vec
        template<class Matrix>
        bool step(vec& x, const Matrix& A)
        {

            // solution already found lol !
            if( !phi ) return true;

            Ap = A(p);
            const real pAp = p.dot(Ap);

            // fail
            if( !pAp ) return false;

            // const real alpha = phi2 / pAp;
            const real alpha = r.dot(p) / pAp;
            
            x += alpha * p;
            r -= alpha * Ap;

            const real old = phi2;

            phi2 = r.squaredNorm();
            phi = std::sqrt( phi2 );
            const real mu = phi2 / old;

            p = r + mu * p;
            ++k;

            return true;
        }


    };


};



#endif
