#ifndef COMPLIANT_UTILS_KRYLOV_H
#define COMPLIANT_UTILS_KRYLOV_H

#include <Eigen/Core>

template<class U>
struct krylov
{

    // some useful types
    typedef U real;
    typedef Eigen::Matrix<real, Eigen::Dynamic, 1> vec;

    struct params
    {
        params() : iterations(0), precision(0), restart(0) { }
        unsigned iterations;
        real precision;
        unsigned restart;
    };


    // by default, write a 'vec' as a line (a column is not readable)
    friend std::ostream& operator<<( std::ofstream& o, const vec& v )
    {
        return o<<v.transpose();
    }


};


#endif
