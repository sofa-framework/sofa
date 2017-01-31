#ifndef SOFA_HELPER_STABLE_VECTOR_H
#define SOFA_HELPER_STABLE_VECTOR_H


#include <boost/container/stable_vector.hpp>


namespace sofa
{
namespace helper
{

    template<class T, class A = std::allocator<T>>
    using stable_vector = boost::container::stable_vector<T,A>;


} // namespace helper
} // namespace sofa

#endif
