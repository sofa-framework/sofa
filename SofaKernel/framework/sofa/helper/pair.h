#ifndef SOFA_HELPER_PAIR_H
#define SOFA_HELPER_PAIR_H

#include <sofa/helper/helper.h>

#include <utility>
#include <iostream>
#include <string>


/// adding string serialization to std::pair to make it compatible with Data
/// \todo: refactoring of the containers required
/// More info PR #113: https://github.com/sofa-framework/sofa/pull/113


namespace std
{

/// Output stream
template<class T1, class T2>
std::ostream& operator<< ( std::ostream& o, const std::pair<T1,T2>& p )
{
    return o << p.first << " " << p.second;
}

/// Input stream
template<class T1, class T2>
std::istream& operator>> ( std::istream& in, std::pair<T1,T2>& p )
{
    return in >> p.first >> p.second;
}


} // namespace std

#endif
