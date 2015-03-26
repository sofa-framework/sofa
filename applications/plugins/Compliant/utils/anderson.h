#ifndef COMPLIANT_UTILS_ANDERSON_H
#define COMPLIANT_UTILS_ANDERSON_H

#include "eigen_types.h"
#include <Eigen/Cholesky>

namespace utils {

// anderson(m)
class anderson : public eigen_types {

    unsigned n, m;
    dmat G, F, K;

    unsigned k;

    vec metric;

    // work stuff
    vec old, alpha;
    Eigen::LDLT<dmat> inv;
    
public:

    // n = dimension, m = history size, metric = diagonal inner
    // product for least-squares
    anderson(unsigned n, unsigned m = 2, const vec& metric = vec() );

    // apply acceleration to fixed-point vector
    void operator()(vec& x);

};


}


#endif
