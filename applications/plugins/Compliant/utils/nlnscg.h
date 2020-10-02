#ifndef COMPLIANT_UTILS_NLNSCG_H
#define COMPLIANT_UTILS_NLNSCG_H

#include "eigen_types.h"
#include <Eigen/Cholesky>

namespace utils {

// nlnscg(m)
class nlnscg : public eigen_types {

//    const unsigned n;
    const vec metric;
    
    unsigned k;
    
    vec old, g, p;
    real g2;


public:

    // n = dimension, metric = diagonal inner product for
    // least-squares
    nlnscg(unsigned n, const vec& metric = vec() );
    
    // apply acceleration to fixed-point vector
    void operator()(vec& x);

};


}


#endif
