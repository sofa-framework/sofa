#ifndef SOFA_HELPER_PAIR_H
#define SOFA_HELPER_PAIR_H

#include <sofa/helper/helper.h>

#include <utility>
#include <iostream>
#include <string>


/// adding string serialization to std::pair to make it compatible with Data
/// \todo: refactoring of the containers required
/// More info PR #113: https://github.com/sofa-framework/sofa/pull/113


namespace sofa
{
namespace helper
{
using std::pair ;

/// Output stream
template<class T1, class T2>
std::ostream& operator<< ( std::ostream& o, const sofa::helper::pair<T1,T2>& p )
{
    return o << p.first << " " << p.second;
}

/// Input stream
template<class T1, class T2>
std::istream& operator>> ( std::istream& in, sofa::helper::pair<T1,T2>& p )
{
    return in >> p.first >> p.second;
}

} // namespace helper
} // namespace sofa

#endif
