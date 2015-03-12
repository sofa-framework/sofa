#include "SubKKT.h"
#include "Response.h"


namespace sofa {
namespace component {
namespace linearsolver {


static SubKKT::mat projection_basis(const SubKKT::mat& P) {
    SubKKT::mat res(P.rows(), P.nonZeros());

    unsigned off = 0;
    for(unsigned i = 0, n = P.rows(); i < n; ++i) {

        res.startVec(i);

        for(SubKKT::mat::InnerIterator it(P, i); it; ++it) {
            if( it.value() ) {
                res.insertBack(i, off) = it.value();
            }
            
            ++off;
        }

    }

    return res;
}

SubKKT SubKKT::projected_primal(const AssembledSystem& sys) {
    SubKKT res;

    // matrix P conveniently filters out 
    res.P = projection_basis(sys.P);

    // TODO optimize ?
    res.A = res.P.transpose() * sys.H * res.P;

    return res;
}


SubKKT::SubKKT() { }


void SubKKT::factor(Response& resp) const {
    resp.factor(A);
}


void SubKKT::solve(Response& resp,
                   vec& res,
                   const vec& rhs) const {
    assert( rhs.size() == size_full() );
    res.resize( size_full() );

    tmp1.resize( size_sub() );

    if( P.cols() ) {
        tmp1.head(P.cols()).noalias() = P.transpose() * rhs.head(P.rows());
    }
    if( Q.cols() ) {
        tmp1.tail(Q.cols()).noalias() = Q.transpose() * rhs.tail(Q.rows());
    }
    
    // system solve
    resp.solve(tmp2, tmp1);

    // remap
    if( P.cols() ) {
        res.head(P.rows()).noalias() = P * tmp2.head(P.cols());
    }
    
    if( Q.cols() ) {
        res.head(Q.rows()).noalias() = Q * tmp2.tail(Q.cols());
    }

}


unsigned SubKKT::size_full() const {
    return P.rows() + Q.rows();
}

unsigned SubKKT::size_sub() const {
    return P.cols() + Q.cols();
}


void SubKKT::solve(Response& resp,
                   cmat& res,
                   const cmat& rhs) const {
    res.resize(rhs.rows(), rhs.cols());
    
    if( Q.cols() ) {
        throw std::logic_error("sorry, not implemented");
    }
    
    cmat tmp1, tmp2;
    
    tmp1 = P.transpose() * rhs;
    resp.solve(tmp2, tmp1);
    
    res = P * tmp2;
}



}
}
}
