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
#ifndef SOFA_HELPER_deque_H
#define SOFA_HELPER_deque_H

#include <sofa/helper/config.h>

#include <sofa/type/hardening.h>

#include <deque>
#include <string>
#include <iostream>
#include <cstdlib>
#include <cerrno>
#include <climits>

#include <sofa/helper/logging/Messaging.h>

/// adding string serialization to std::deque to make it compatible with Data
/// \todo: refactoring of the containers required
/// More info PR #113: https://github.com/sofa-framework/sofa/pull/113


namespace std
{

/// Output stream
template<class T>
std::ostream& operator<< ( std::ostream& os, const std::deque<T>& d )
{
    if( d.size()>0 )
    {
        for( unsigned int i=0, iend=d.size()-1; i<iend; ++i ) os<<d[i]<<" ";
        os<<d.back();
    }
    return os;
}

/// Input stream
template<class T>
std::istream& operator>> ( std::istream& in, std::deque<T>& d )
{
    T t=T();
    d.clear();
    while(in>>t)
        d.push_back(t);
    if( in.rdstate() & std::ios_base::eofbit ) { in.clear(); }
    return in;
}







/// Input stream
/// Specialization for reading dequeues of int and unsigned int using "A-B" notation for all integers between A and B, optionally specifying a step using "A-B-step" notation.
template<>
inline std::istream& operator>>( std::istream& in, std::deque<int>& d )
{
    int t;
    d.clear();
    std::string s;
    while(in>>s)
    {
        const std::string::size_type hyphen = s.find_first_of('-',1);
        if (hyphen == std::string::npos)
        {
            if (!sofa::type::hardening::safeStrToInt(s, t))
            {
                msg_error("deque") << "parsing \""<<s<<"\": invalid integer";
                continue;
            }
            d.push_back(t);
        }
        else
        {
            int t1,t2,tinc;
            std::string s1(s,0,hyphen);
            if (!sofa::type::hardening::safeStrToInt(s1, t1))
            {
                msg_error("deque") << "parsing \""<<s<<"\": invalid integer";
                continue;
            }
            const std::string::size_type hyphen2 = s.find_first_of('-',hyphen+2);
            if (hyphen2 == std::string::npos)
            {
                std::string s2(s,hyphen+1);
                if (!sofa::type::hardening::safeStrToInt(s2, t2))
                {
                    msg_error("deque") << "parsing \""<<s<<"\": invalid integer";
                    continue;
                }
                tinc = (t1<t2) ? 1 : -1;
            }
            else
            {
                std::string s2(s,hyphen+1,hyphen2-hyphen-1);
                std::string s3(s,hyphen2+1);
                if (!sofa::type::hardening::safeStrToInt(s2, t2) || !sofa::type::hardening::safeStrToInt(s3, tinc))
                {
                    msg_error("deque") << "parsing \""<<s<<"\": invalid integer";
                    continue;
                }
                if (tinc == 0)
                {
                    msg_error("deque") << "parsing \""<<s<<"\": increment is 0";
                    tinc = (t1<t2) ? 1 : -1;
                }
                if ((t2-t1)*tinc < 0)
                {
                    // increment not of the same sign as t2-t1 : swap t1<->t2
                    t = t1;
                    t1 = t2;
                    t2 = t;
                }
            }
            if (tinc < 0)
                for (t=t1; t>=t2; t+=tinc)
                    d.push_back(t);
            else
                for (t=t1; t<=t2; t+=tinc)
                    d.push_back(t);
        }
    }
    if( in.rdstate() & std::ios_base::eofbit ) { in.clear(); }
    return in;
}

/// Output stream
/// Specialization for writing dequeues of unsigned char
template<>
inline std::ostream& operator<<(std::ostream& os, const std::deque<unsigned char>& d)
{
    if( d.size()>0 )
    {
        unsigned int i=0, iend=d.size()-1;
        for( ; i<iend; ++i ) os<<(int)d[i]<<" ";
        os<<(int)d[iend];
    }
    return os;
}

/// Input stream
/// Specialization for writing dequeues of unsigned char
template<>
inline std::istream& operator>>(std::istream& in, std::deque<unsigned char>& d)
{
    int t;
    d.clear();
    while(in>>t)
    {
        d.push_back((unsigned char)t);
    }
    if( in.rdstate() & std::ios_base::eofbit ) { in.clear(); }
    return in;
}

/// Input stream
/// Specialization for reading dequeues of int and unsigned int using "A-B" notation for all integers between A and B
template<>
inline std::istream& operator>>( std::istream& in, std::deque<unsigned int>& d )
{
    unsigned int t;
    d.clear();
    std::string s;
    while(in>>s)
    {
        const std::string::size_type hyphen = s.find_first_of('-',1);
        if (hyphen == std::string::npos)
        {
            if (!sofa::type::hardening::safeStrToUInt(s, t))
            {
                msg_error("deque") << "parsing \""<<s<<"\": invalid unsigned integer";
                continue;
            }
            d.push_back(t);
        }
        else
        {
            unsigned int t1,t2;
            int tinc;
            std::string s1(s,0,hyphen);
            if (!sofa::type::hardening::safeStrToUInt(s1, t1))
            {
                msg_error("deque") << "parsing \""<<s<<"\": invalid unsigned integer";
                continue;
            }
            const std::string::size_type hyphen2 = s.find_first_of('-',hyphen+2);
            if (hyphen2 == std::string::npos)
            {
                std::string s2(s,hyphen+1);
                if (!sofa::type::hardening::safeStrToUInt(s2, t2))
                {
                    msg_error("deque") << "parsing \""<<s<<"\": invalid unsigned integer";
                    continue;
                }
                tinc = (t1<t2) ? 1 : -1;
            }
            else
            {
                std::string s2(s,hyphen+1,hyphen2-hyphen-1);
                std::string s3(s,hyphen2+1);
                if (!sofa::type::hardening::safeStrToUInt(s2, t2) || !sofa::type::hardening::safeStrToInt(s3, tinc))
                {
                    msg_error("deque") << "parsing \""<<s<<"\": invalid integer";
                    continue;
                }
                if (tinc == 0)
                {
                    msg_error("deque") << "parsing \""<<s<<"\": increment is 0";
                    tinc = (t1<t2) ? 1 : -1;
                }
                if (((int)(t2-t1))*tinc < 0)
                {
                    // increment not of the same sign as t2-t1 : swap t1<->t2
                    t = t1;
                    t1 = t2;
                    t2 = t;
                }
            }
            if (tinc < 0)
                for (t=t1; t>=t2; t=(unsigned int)((int)t+tinc))
                    d.push_back(t);
            else
                for (t=t1; t<=t2; t=(unsigned int)((int)t+tinc))
                    d.push_back(t);
        }
    }
    if( in.rdstate() & std::ios_base::eofbit ) { in.clear(); }
    return in;
}

} // namespace std

#endif
