#ifndef SOFA_HELPER_FIXED_ARRAY_ALGORITHMS_H
#define SOFA_HELPER_FIXED_ARRAY_ALGORITHMS_H

#include <sofa/helper/helper.h>
#include <sofa/helper/rmath.h>
#include <sofa/helper/fixed_array.h>
namespace sofa
{

namespace helper
{

namespace pairwise
{

/// @brief clamp a single value. This function should be removed when std::clamp will be available
template<class T>
const T& stdclamp( const T& v, const T& lo, const T& hi )
{
    assert( !(hi < lo) );
    return (v < lo) ? lo : (hi < v) ? hi : v;
}

/// @brief clamp all the values of a fixed_array to be within a given interval.
template<class T, class TT=typename T::value_type, size_t TN=T::static_size>
T clamp(const T& in, const TT& minValue, const TT& maxValue)
{
    T result {};
    for(std::size_t i=0; i < TN; ++i)
    {
        result[i] = stdclamp(in[i], minValue, maxValue);
    }
    return result;
}

/// @brief pairwise add of two fixed_array
template<class T, class TT=typename T::value_type, size_t TN=T::static_size>
T operator+(const T& l, const T& r)
{
    T result {};
    for(std::size_t i=0; i < TN; ++i)
    {
        result[i] = l[i] + r[i];
    }
    return result;
}

/// @brief pairwise subtract of two fixed_array
template<class T, class TT=typename T::value_type, size_t TN=T::static_size>
T operator-(const T& l, const T& r)
{
    T result {};
    for(std::size_t i=0; i < TN; ++i)
    {
        result[i] = l[i] - r[i];
    }
    return result;
}

/// @brief multiply from l the r components.
template<class T, class TT=typename T::value_type, size_t TN=T::static_size>
T operator*(const T& r, const typename T::value_type& f)
{
    T result {};
    for(std::size_t i=0; i < TN; ++i)
    {
        result[i] = r[i] * f;
    }
    return result;
}

/// @brief multiply from l the r components.
template<class T, class TT=typename T::value_type, size_t TN=T::static_size>
T operator/(const T& r, const typename T::value_type& f)
{
    T result {};
    for(std::size_t i=0; i < TN; ++i)
    {
        result[i] = r[i] / f;
    }
    return result;
}


} /// namespace pairwise

} /// namespace helper

} /// namespace sofa

#endif
