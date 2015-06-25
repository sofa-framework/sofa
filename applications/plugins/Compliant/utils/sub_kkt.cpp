#include "sub_kkt.inl"
#include "scoped.h"

#include "../assembly/AssembledSystem.h"

namespace utils {


struct sub_kkt::helper {

    // TODO projection_basis from a vec mask

    typedef rmat::InnerIterator iterator;
        
    // build a projection basis based on filtering matrix P
    // P must be diagonal with 0, 1 on the diagonal
    static void projection_basis(rmat& res, const rmat& P) {

        res.resize(P.rows(), P.nonZeros());
        res.reserve(P.nonZeros());
    
        unsigned off = 0;
        for(unsigned i = 0, n = P.rows(); i < n; ++i) {

            res.startVec(i);

            if( iterator(P, i) ) {
                res.insertBack(i, off++) = 1;
            }

        }
    
        res.finalize();
    }


    

    // builds the projected primal system P^T H P, where P is the primal
    // projection basis
    static void filter_primal(rmat& res,
                              const rmat& H,
                              const rmat& primal) {
        
        res.resize(primal.cols(), primal.cols());
        res.reserve(H.nonZeros());

        for(unsigned i = 0, n = primal.rows(); i < n; ++i) {
            for( iterator it(primal, i); it; ++it) {
                // we have a non-zero row in P, hence in res at row
                // it.col()
                res.startVec(it.col());

                for( iterator itH(H, i); itH; ++itH) {
                
                    for(iterator it2(primal, itH.col()); it2; ++it2) {
                        // we have a non-zero row in P, non-zero col in
                        // res at col it2.col()
                        res.insertBack(it.col(), it2.col()) = itH.value();
                    }
                
                }
            
            
            }
        
        }
        res.finalize();
    }
    
    // build a projected KKT system based on primal projection matrix P
    // and dual projection matrix Q (to only include bilateral constraints)
    static void filter_kkt(rmat& res,
                           
                           const system_type& sys,
                           
                           const rmat& primal,
                           const rmat& dual,
                           
                           real eps,

                           bool only_lower = false) {

        const unsigned size_sub = primal.cols() + dual.cols();
        
        res.resize(size_sub, size_sub);
        res.reserve(sys.H.nonZeros() + 2 * sys.J.nonZeros() + sys.C.nonZeros() );

        rmat JT;

        if( !only_lower ) {
            JT = sys.J.transpose();
        }
        
        const unsigned primal_sub = primal.cols();
        const unsigned primal_full = primal.rows();        

        for(unsigned i = 0; i < primal_full; ++i) {
            
            const iterator has_row(primal, i);

            if( !has_row ) continue;

            const unsigned sub_row = has_row.col();
            res.startVec( sub_row );
            
            // H
            for(iterator itH(sys.H, i); itH; ++itH) {
                
                if(only_lower && itH.col() > itH.row()) break;
                
                const iterator has_row(primal, itH.col());

                if( !has_row ) continue;
                const unsigned sub_col = has_row.col();
                
                res.insertBack( sub_row, sub_col) = itH.value();
            }

            if( only_lower ) continue;

            // JT
            for(iterator itJT(JT, i); itJT; ++itJT) {

                const iterator has_row(primal, itJT.col());

                if( !has_row ) continue;
                const unsigned sub_col = has_row.col();
                
                res.insertBack(sub_row, primal_sub + sub_col) = -itJT.value();
            }

        }

        for(unsigned i = 0; i < sys.n; ++i) {

            const iterator has_row(dual, i);

            if( !has_row ) continue;
            const unsigned sub_row = primal_sub + has_row.col();

            res.startVec(sub_row);
            
            // J
            for(iterator itJ(sys.J, i); itJ; ++itJ) {

                const iterator has_row(primal, itJ.col());

                if( !has_row ) continue;

                const unsigned sub_col = has_row.col();
                
                res.insertBack(sub_row, sub_col) = -itJ.value();
            }
            
            // C
            real* diag = 0;
            for(iterator itC(sys.C, i); itC; ++itC) {

                const unsigned sub_col = primal_sub + itC.col();
                
                real& ref = res.insertBack(sub_row, sub_col);
                ref = -itC.value();
                
                // store ref on diagonal element
                if(itC.col() == itC.row()) diag = &ref;

            }

            // we did not encounter diagonal in C, we need to create it
            if( !diag && eps ) {
                real& ref = res.insertBack(sub_row, sub_row);
                ref = 0;
                diag = &ref;
            }
            
            if( eps ) *diag = std::min(-eps, *diag);
            // if( eps ) *diag -= eps;
        }

        res.finalize();
    }



   
    

};


void sub_kkt::projected_primal(const system_type& sys) {
    scoped::timer step("subsystem projection");
    
    helper::projection_basis(primal, sys.P);
    dual = rmat();
    
    helper::filter_primal(matrix, sys.H, primal);
}



void sub_kkt::projected_kkt(const system_type& sys, real eps, bool only_lower) {
    scoped::timer step("subsystem projection");

    helper::projection_basis(primal, sys.P);

    if( sys.n ) {
        dual.resize(sys.n, sys.n);
        dual.setIdentity();
        
        helper::filter_kkt(matrix, sys, primal, dual, eps, only_lower);
    } else { 
        dual = rmat();
        
        helper::filter_primal(matrix, sys.H, primal);
    }
}

struct sub_kkt::prod_action {
    const rmat& matrix;
    const bool only_lower;
    
    prod_action(const rmat& matrix, bool only_lower)
        : matrix(matrix),
          only_lower(only_lower) { }

    void operator()(vec& res, const vec& rhs) const {
        if(only_lower) { 
            res.noalias() = matrix.selfadjointView<Eigen::Lower>() * rhs;
        } else {
            res.noalias() = matrix * rhs;
        }
    }

};


void sub_kkt::prod(vec& res, const vec& rhs, bool only_lower) const {

    project_unproject(prod_action(matrix, only_lower), res, rhs);
 
}





unsigned sub_kkt::size_full() const {
    return primal.rows() + dual.rows();
}

unsigned sub_kkt::size_sub() const {
    return primal.cols() + dual.cols();
}




}
