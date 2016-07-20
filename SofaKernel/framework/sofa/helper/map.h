/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_HELPER_MAP_H
#define SOFA_HELPER_MAP_H

#include <sofa/helper/helper.h>

#include <map>
#include <iostream>
#include <sstream>
#include <string>

namespace std
{

/// Output stream
template<class K, class T>
std::ostream& operator<< ( std::ostream& o, const std::map<K,T>& m )
{
    typename std::map<K,T>::const_iterator it=m.begin();
    if (it == m.end()) return o;
    o << it->first << " " << it->second; it++;
    for ( ; it != m.end() ; ++it)
    {
        o << "\n" << it->first << " " << it->second;
    }

    return o;
}

/// Input stream
template<class K, class T>
std::istream& operator>> ( std::istream& i, std::map<K,T>& m )
{
    m.clear();
    std::string line;
    while (!i.eof())
    {
        K k; T t;
        i >> k;
        std::getline(i,line);
        if (line.empty()) break;
        std::istringstream li(line);
        li >> t;
        m[k] = t;
    }
    return i;
}

}

namespace sofa
{

namespace helper
{

} // namespace helper

} // namespace sofa

#endif
