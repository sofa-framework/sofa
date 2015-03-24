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
                   const vec& rhs) const {

    res.resize( size_full() );

    solve_filtered( resp, vtmp2, rhs );

    // remap
    if( P.cols() ) {
        res.head(P.rows()).noalias() = P * vtmp2.head(P.cols());
    }
    
    if( Q.cols() ) {
        res.tail(Q.rows()).noalias() = Q * vtmp2.tail(Q.cols());
    }

}



template< class Solver >
void SubKKT::solve_filtered(const Solver& resp,
                   vec& res,
                   const vec& rhs) const {
    assert( rhs.size() == size_full() );
    res.resize( size_full() );

    vtmp1.resize( size_sub() );
    res.resize( size_sub() );

    if( P.cols() ) {
        vtmp1.head(P.cols()).noalias() = P.transpose() * rhs.head(P.rows());
    }
    if( Q.cols() ) {
        vtmp1.tail(Q.cols()).noalias() = Q.transpose() * rhs.tail(Q.rows());
    }

    // system solve
    resp.solve(res, vtmp1);
}




}
}
}

#endif
