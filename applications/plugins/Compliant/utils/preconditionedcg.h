#ifndef COMPLIANT_UTILS_PRECONDITIONEDCG_H
#define COMPLIANT_UTILS_PRECONDITIONEDCG_H

#include <cassert>
#include "krylov.h"


template<class U>
struct preconditionedcg
{

    typedef ::krylov<U> krylov;

    typedef typename krylov::vec vec;
    typedef typename krylov::real real;
    typedef typename krylov::natural natural;
    typedef typename krylov::params params;

    // solves Ax = b using a preconditioned conjugate gradient.
    // @A is a function object vec -> vec implementing matrix multiplication and approximated inverse multiplication (preconditioner)
    template<class Matrix, class Preconditioner>
    static void solve(vec& x, const Matrix& A, const Preconditioner& P, const vec& b, params& p)
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
        action d( residual );
        d.init( P );

        natural i;
        for( i = 0; i < p.iterations && d.phi > p.precision; ++i )
        {
            d.step(x, A, P);
        }
        p.iterations = i;

    }


    // contains all the data needed for minres iterations
    struct action
    {

        vec p;			// descent direction
        vec& r;			// residual
        vec z;			// preconditioned residual
        vec Ap;			// A(p)

        real phi2;		// residual squared norm
        real phi;			// residual norm

        natural k;		// iteration

        action( vec& r ) : r(r) {}

        // initializes minres given initial residual @r
        template<class Preconditioner>
        void init(const Preconditioner& P)
        {
            z = P(r);
            p = r;
            phi2 = r.dot(z);
            phi = std::sqrt( phi2 );
            k = 1;
        }


        // performs one cg step. returns false on singularity, true
        // otherwise. @A is a function object vec -> vec
        template<class Matrix, class Preconditioner>
        bool step(vec& x, const Matrix& A, const Preconditioner& P)
        {
            Ap = A(p);
            const real pAp = p.dot(Ap);

            // fail
            if( !pAp ) return false;

            // const real alpha = phi2 / pAp;
            const real alpha = r.dot(z) / pAp;
            
            x += alpha * p;
            r -= alpha * Ap;
            z = P(r);

            const real old = phi2;

            phi2 = r.dot(z);
            phi = std::sqrt( phi2 );
            const real mu = phi2 / old;

            p = z + mu * p;
            ++k;

            return true;
        }


    };


};



#endif
