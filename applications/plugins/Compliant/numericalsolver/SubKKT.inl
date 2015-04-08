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

    const size_t sub_size = size_sub();

    // project
    vtmp1.resize( sub_size );
    if( P.cols() ) {
        vtmp1.head(P.cols()).noalias() = P.transpose() * rhs.head(P.rows());
    }
    if( Q.cols() ) {
        vtmp1.tail(Q.cols()).noalias() = Q.transpose() * rhs.tail(Q.rows());
    }

    // system solve
    vtmp2.resize( sub_size );
    resp.solve(vtmp2, vtmp1);

    // remap
    res.resize( rhs.size() );
    if( P.cols() ) {
        res.head(P.rows()).noalias() = P * vtmp2.head(P.cols());
    }
    if( Q.cols() ) {
        res.tail(Q.rows()).noalias() = Q * vtmp2.tail(Q.cols());
    }

}





}
}
}

#endif
