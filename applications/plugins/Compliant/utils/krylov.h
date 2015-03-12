#ifndef COMPLIANT_UTILS_KRYLOV_H
#define COMPLIANT_UTILS_KRYLOV_H

#include <Eigen/Core>

template<class U>
struct krylov
{

    // some useful types
    typedef U real;
    typedef Eigen::Matrix<real, Eigen::Dynamic, 1> vec;
    typedef unsigned int natural;

    struct params
    {
        params() : iterations(0), precision(0), restart(0) { }
        natural iterations;
        real precision;
        unsigned restart;
    };


};


#endif
