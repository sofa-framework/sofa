#include "SubKKT.h"
#include "Response.h"

#include "../utils/scoped.h"
#include "../utils/sparse.h"

namespace sofa {
namespace component {
namespace linearsolver {


typedef SubKKT::rmat rmat;

// P must be diagonal with 0, 1 on the diagonal
static void projection_basis(rmat& res, const rmat& P, bool* is_identity) {
    res.resize(P.rows(), P.nonZeros());
    res.setZero();
    
    unsigned off = 0;
    for(unsigned i = 0, n = P.rows(); i < n; ++i) {

        res.startVec(i);

        for(rmat::InnerIterator it(P, i); it; ++it) {
            if( it.value() ) {
                res.insertBack(i, off) = it.value();
            }
            
            ++off;
        }

    }
    res.finalize();
    *is_identity = (off == P.rows() );
}


static void filter_primal(rmat& res, const rmat& H, const rmat& P) {
    res.resize(P.cols(), P.cols());
    res.setZero();
    res.reserve(H.nonZeros());
    
    for(unsigned i = 0, n = P.rows(); i < n; ++i) {
        for(rmat::InnerIterator it(P, i); it; ++it) {
            // we have a non-zero row in P, hence in res at row
            // it.col()
            res.startVec(it.col());

            for(rmat::InnerIterator itH(H, i); itH; ++itH) {
                
                for(rmat::InnerIterator it2(P, itH.col()); it2; ++it2) {
                    // we have a non-zero row in P, non-zero col in
                    // res at col it2.col()
                    res.insertBack(it.col(), it2.col()) = itH.value();
                }
                
            }
            
            
        }
        
    }
    res.finalize();
}


static void filter_kkt(rmat& res, const rmat& H, const rmat& P, const rmat& J, const rmat& C) {
    res.resize(P.cols() + J.rows(), P.cols() + J.rows());
    res.setZero();

    res.reserve(H.nonZeros() + 2 * J.nonZeros() + C.nonZeros());
    
    const rmat JT = J.transpose();
    
    for(unsigned i = 0, n = P.rows(); i < n; ++i) {
        for(rmat::InnerIterator it(P, i); it; ++it) {
            // we have a non-zero row in P, hence in res at row
            // it.col()
            res.startVec(it.col());

            for(rmat::InnerIterator itH(H, i); itH; ++itH) {
                
                for(rmat::InnerIterator it2(P, itH.col()); it2; ++it2) {
                    // we have a non-zero row in P, non-zero col in
                    // res at col it2.col()
                    res.insertBack(it.col(), it2.col()) = itH.value();
                }
                
            }

            for(rmat::InnerIterator itJT(JT, i); itJT; ++itJT) {
                res.insertBack(it.col(), P.cols() + itJT.col()) = -itJT.value();
            }
            
        }
        
    }

    for(unsigned i = 0, n = J.rows(); i < n; ++i) {
        res.startVec(P.cols() + i);
        
        for(rmat::InnerIterator itJ(J, i); itJ; ++itJ) {
            for(rmat::InnerIterator it2(P, itJ.col()); it2; ++it2) {
                // we have a non-zero row in P, so non-zero col in
                // res at col it2.col()
                res.insertBack(P.cols() + i, it2.col()) = -itJ.value();
            }
        }

        for(rmat::InnerIterator itC(C, i); itC; ++itC) {
            res.insertBack(P.cols() + i, P.cols() + itC.col()) = -itC.value();
        }
    }


    res.finalize();
}



void SubKKT::projected_primal(SubKKT& res, const AssembledSystem& sys) {
    scoped::timer step("subsystem projection");
    
    // matrix P conveniently filters out
    bool identity;
    projection_basis(res.P, sys.P, &identity);

    // TODO optimize
    filter_primal(res.A, sys.H, res.P);

    res.Q = rmat();
}


void SubKKT::projected_kkt(SubKKT& res, const AssembledSystem& sys) {
    scoped::timer step("subsystem projection");

    bool identity;
    projection_basis(res.P, sys.P, &identity);

    if(sys.n) {
        filter_kkt(res.A, sys.H, res.P, sys.J, sys.C);

        res.Q.resize(sys.n, sys.n);
        res.Q.setIdentity();
    } else {
        filter_primal(res.A, sys.H, res.P);
        res.Q = rmat();
    }

}


SubKKT::SubKKT() { }


void SubKKT::factor(Response& resp) const {
    scoped::timer step("subsystem factor");
    resp.factor(A);
}


void SubKKT::solve(const Response& resp,
                   vec& res,
                   const vec& rhs) const {
    assert( rhs.size() == size_full() );
    res.resize( size_full() );

    vtmp1.resize( size_sub() );
    vtmp2.resize( size_sub() );    

    if( P.cols() ) {
        vtmp1.head(P.cols()).noalias() = P.transpose() * rhs.head(P.rows());
    }
    if( Q.cols() ) {
        vtmp1.tail(Q.cols()).noalias() = Q.transpose() * rhs.tail(Q.rows());
    }
    
    // system solve
    resp.solve(vtmp2, vtmp1);

    // remap
    if( P.cols() ) {
        res.head(P.rows()).noalias() = P * vtmp2.head(P.cols());
    }
    
    if( Q.cols() ) {
        res.tail(Q.rows()).noalias() = Q * vtmp2.tail(Q.cols());
    }

}


unsigned SubKKT::size_full() const {
    return P.rows() + Q.rows();
}

unsigned SubKKT::size_sub() const {
    return P.cols() + Q.cols();
}


void SubKKT::solve(const Response& resp,
                   cmat& res,
                   const cmat& rhs) const {
    res.resize(rhs.rows(), rhs.cols());
    
    if( Q.cols() ) {
        throw std::logic_error("sorry, not implemented");
    }
    
    mtmp1 = P.transpose() * rhs;
        
    resp.solve(mtmp2, mtmp1);

    // mtmp3 = P;

    // not sure if this causes a temporary
    res = P * mtmp2;
}

void SubKKT::solve_opt(const Response& resp,
                       cmat& res,
                       const rmat& rhs) const {
    res.resize(rhs.rows(), rhs.cols());
    
    if( Q.cols() ) {
        throw std::logic_error("sorry, not implemented");
    }

    sparse::fast_prod(mtmp1, P.transpose(), rhs.transpose());
    // mtmp1 = P.transpose() * rhs.transpose();
    
    resp.solve(mtmp2, mtmp1);
    mtmp3 = P;

    // not sure if this causes a temporary
    sparse::fast_prod(res, mtmp3, mtmp2);
    // res = mtmp3 * mtmp2;
}



}
}
}
