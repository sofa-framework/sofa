#include "nlnscg.h"

namespace utils {
nlnscg::nlnscg(unsigned n, const vec& metric)
    : /*n(n),*/ metric(metric), k(0), p( vec::Zero(n) ), g2(0) {

    
}


void nlnscg::operator()(vec& x) {
    
    if( k > 0 ) {
        g = old - x;

        const real tmp = metric.size() ? g.dot( metric.cwiseProduct(g) ) :  g.dot(g);

        real beta = 0;

        if(g2) {
            beta = tmp / g2;

            if( beta <= 1 ) {

                x += beta * p;
                p = beta * p - g;
                
            } else {
                p.setZero();
            }
            
        }
        
        g2 = tmp;
    }
     
    old = x;
    ++k;
}


}
