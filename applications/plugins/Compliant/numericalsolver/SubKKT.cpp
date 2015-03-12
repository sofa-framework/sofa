#include "SubKKT.h"
#include "Response.h"


namespace sofa {
namespace component {
namespace linearsolver {


SubKKT::mat SubKKT::projection_basis(const mat& P) {
    mat res(P.rows(), P.nonZeros());

    unsigned off = 0;
    for(unsigned i = 0, n = P.rows(); i < n; ++i) {

        res.startVec(i);

        for(mat::InnerIterator it(P, i); it; ++it) {
            if( it.value() ) {
                res.insertBack(i, off) = it.value();
            }
            
            ++off;
        }

    }
    
}

// TODO optimize ?
SubKKT::SubKKT(const AssembledSystem& sys):
    P(projection_basis(sys.P)),
    A(P.transpose() * sys.H * P) {


}


void SubKKT::factor(Response& resp) const { resp.factor(A); }


// TODO we should also use Q here
void SubKKT::solve(Response& resp,
                   vec& res,
                   const vec& rhs) const {

    tmp1.noalias() = P.transpose() * rhs;
    resp.solve(tmp2, tmp1);
    res.noalias() = P * tmp2;

}


void SubKKT::solve(Response& resp,
                   cmat& res,
                   const cmat& rhs) const {
    cmat tmp1, tmp2;
    
    tmp1 = P.transpose() * rhs;
    resp.solve(tmp2, tmp1);
    
    res = P * tmp2;
}



}
}
}
