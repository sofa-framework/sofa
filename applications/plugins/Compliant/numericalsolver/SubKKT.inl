#ifndef COMPLIANT_SUBKKT_INL
#define COMPLIANT_SUBKKT_INL

#include "SubKKT.h"
#include "../utils/scoped.h"
#include "../utils/sparse.h"
#include "Response.h"

namespace sofa {
namespace component {
namespace linearsolver {

template< class Solver >
inline void SubKKT::factor(Solver& resp) const {
    scoped::timer step("subsystem factor");
    resp.factor(A);
}



template< class Solver >
void SubKKT::solve(const Solver& resp,
                   vec& res,
                   const vec& rhs,
                   ProblemType problem) const {

    res.resize( rhs.size() );

    solve_filtered( resp, vtmp2, rhs, problem );

    // remap
    if( (problem & PRIMAL) && P.cols() ) {
        res.head(P.rows()).noalias() = P * vtmp2.head(P.cols());
    }
    
    if( (problem & DUAL) && Q.cols() ) {
        res.tail(Q.rows()).noalias() = Q * vtmp2.tail(Q.cols());
    }

}



template< class Solver >
void SubKKT::solve_filtered(const Solver& resp,
                   vec& res,
                   const vec& rhs,
                   ProblemType problem ) const {

    size_t size;

    if( problem == FULL )
    {
        assert( rhs.size()==size_full() );
        size = size_sub();
    }
    else if( problem == PRIMAL )
    {
        assert( rhs.size()==P.rows() );
        size = P.cols();
    }
    else /*DUAL*/
    {
        assert( rhs.size()==Q.rows() );
        size = Q.cols();
    }


    vtmp1.resize( size );
    res.resize( size );

    if( (problem & PRIMAL) && P.cols() ) {
        vtmp1.head(P.cols()).noalias() = P.transpose() * rhs.head(P.rows());
    }
    if( (problem & DUAL) && Q.cols() ) {
        vtmp1.tail(Q.cols()).noalias() = Q.transpose() * rhs.tail(Q.rows());
    }

    // system solve
    resp.solve(res, vtmp1);
}




}
}
}

#endif
