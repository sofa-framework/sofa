/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_HELPER_SET_H
#define SOFA_HELPER_SET_H

#include <sofa/helper/config.h>
#include <set>
#include <iostream>

/// adding string serialization to std::set to make it compatible with Data
/// \todo: refactoring of the containers required
/// More info PR #113: https://github.com/sofa-framework/sofa/pull/113
namespace std
{

/// Output stream
template<class K>
std::ostream& operator<< ( std::ostream& o, const std::set<K>& s )
{
    if( !s.empty() )
    {
        typename std::set<K>::const_iterator i=s.begin(), iend=s.end();
        o << *i;
        ++i;
        for( ; i!=iend; ++i )
            o << ' ' << *i;
    }
    return o;
}

/// Input stream
template<class K>
std::istream& operator>> ( std::istream& i, std::set<K>& s )
{
    K t;
    s.clear();
    while(i>>t)
        s.insert(t);
    if( i.rdstate() & std::ios_base::eofbit ) { i.clear(); }
    return i;
}

/// Input stream
/// Specialization for reading sets of int and unsigned int using "A-B" notation for all integers between A and B, optionnally specifying a step using "A-B-step" notation.
template<> SOFA_HELPER_API std::istream& operator>> ( std::istream& in, std::set<int>& _set );

/// Input stream
/// Specialization for reading sets of int and unsigned int using "A-B" notation for all integers between A and B
template<> SOFA_HELPER_API std::istream& operator>> ( std::istream& in, std::set<unsigned int>& _set );


} // namespace std

#endif
