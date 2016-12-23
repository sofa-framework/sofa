#ifndef SOFA_HELPER_PAIR_H
#define SOFA_HELPER_PAIR_H

#include <sofa/helper/helper.h>

#include <pair>
#include <iostream>
#include <string>


/// adding string serialization to std::list to make it compatible with Data

namespace std
{

/// Output stream
template<class T>
std::ostream& operator<< ( std::ostream& os, const std::pair<T1,T2>& p )
{
    return out << p.first << " " << p.second;
}

/// Input stream
template<class T1, class T2>
std::istream& operator>> ( std::istream& in, std::pair<T1,T2>& p )
{
    return in >> p.first >> p.second;
}


} // namespace std

#endif
