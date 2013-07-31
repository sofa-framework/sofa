#include <rmath.h>
#include <limits>

namespace sofa
{

namespace helper
{


template<> SOFA_HELPER_API
bool isEqual( float x, float y, float threshold )
{
    return rabs(x-y) <= threshold;
}

template<> SOFA_HELPER_API
bool isEqual( double x, double y, double threshold )
{
    return rabs(x-y) <= threshold;
}



} // namespace helper

} // namespace sofa


