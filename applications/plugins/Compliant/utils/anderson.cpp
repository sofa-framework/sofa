#include "anderson.h"

namespace utils {
anderson::anderson(unsigned n, unsigned m, const vec& metric)
    : n(n), m(m), metric(metric) {
    
    if( m ) {
        G = dmat::Zero(n, m);
        F = dmat::Zero(n, m);
        K = dmat::Zero(m, m);
	}

    k = 0;
    
}


void anderson::operator()(vec& x, bool sign_check) {

    if( k > 0 ) {
        const unsigned index = k % m;

        G.col(index) = x;
        F.col(index) = x - old;

        K.col(index).noalias() = F.transpose() * metric.cwiseProduct(F.col(index));
        K.row(index) = K.col(index).transpose();

        inv.compute( K );

        old = x;
        
        if(inv.info() == Eigen::Success) {
        
            // TODO is there a temporary for ones ?
            alpha = inv.solve( vec::Ones(m) );

            alpha /= alpha.array().sum();

            next.noalias() = G * alpha;
            
            if( !sign_check || ((x.array() > 0) == (next.array() > 0)).all() ) {
                x.noalias() = next;
            }
            
        }
    } else {
        old = x;
    }

    ++k;
}


}
