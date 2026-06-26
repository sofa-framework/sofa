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
#include <sofa/helper/config.h>

#include <string>
#include <iostream>
#include <cstdlib>
#include <cerrno>
#include <climits>

#include <sofa/helper/set.h>
#include <sofa/helper/logging/Messaging.h>

/// adding string serialization to std::set to make it compatible with Data
/// \todo: refactoring of the containers required
/// More info PR #113: https://github.com/sofa-framework/sofa/pull/113

namespace
{
    bool safeStrToInt(const std::string& s, int& result)
    {
        char* endptr = nullptr;
        errno = 0;
        long val = std::strtol(s.c_str(), &endptr, 10);
        if (errno != 0 || endptr == s.c_str() || val < INT_MIN || val > INT_MAX)
            return false;
        result = static_cast<int>(val);
        return true;
    }

    bool safeStrToUInt(const std::string& s, unsigned int& result)
    {
        char* endptr = nullptr;
        errno = 0;
        unsigned long val = std::strtoul(s.c_str(), &endptr, 10);
        if (errno != 0 || endptr == s.c_str() || val > UINT_MAX)
            return false;
        result = static_cast<unsigned int>(val);
        return true;
    }
}

namespace std
{

/// Input stream
/// Specialization for reading sets of int and unsigned int using "A-B" notation for all integers between A and B, optionally specifying a step using "A-B-step" notation.
template<>
std::istream& operator>> ( std::istream& in, std::set<int>& _set )
{
    int t;
    _set.clear();
    std::string s;
    while(in>>s)
    {
        const std::string::size_type hyphen = s.find_first_of('-',1);
        if (hyphen == std::string::npos)
        {
            if (!safeStrToInt(s, t))
            {
                msg_error("set") << "parsing \""<<s<<"\": invalid integer";
                continue;
            }
            _set.insert(t);
        }
        else
        {
            int t1,t2,tinc;
            std::string s1(s,0,hyphen);
            if (!safeStrToInt(s1, t1))
            {
                msg_error("set") << "parsing \""<<s<<"\": invalid integer";
                continue;
            }
            const std::string::size_type hyphen2 = s.find_first_of('-',hyphen+2);
            if (hyphen2 == std::string::npos)
            {
                std::string s2(s,hyphen+1);
                if (!safeStrToInt(s2, t2))
                {
                    msg_error("set") << "parsing \""<<s<<"\": invalid integer";
                    continue;
                }
                tinc = (t1<t2) ? 1 : -1;
            }
            else
            {
                std::string s2(s,hyphen+1,hyphen2-hyphen-1);
                std::string s3(s,hyphen2+1);
                if (!safeStrToInt(s2, t2) || !safeStrToInt(s3, tinc))
                {
                    msg_error("set") << "parsing \""<<s<<"\": invalid integer";
                    continue;
                }
                if (tinc == 0)
                {
                    msg_error("set") << "parsing \""<<s<<"\": increment is 0";
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
                    _set.insert(t);
            else
                for (t=t1; t<=t2; t+=tinc)
                    _set.insert(t);
        }
    }
    if( in.rdstate() & std::ios_base::eofbit ) { in.clear(); }
    return in;
}

/// Input stream
/// Specialization for reading sets of int and unsigned int using "A-B" notation for all integers between A and B
template<>
std::istream& operator>> ( std::istream& in, std::set<unsigned int>& _set )
{
    unsigned int t;
    _set.clear();
    std::string s;
    while(in>>s)
    {
        const std::string::size_type hyphen = s.find_first_of('-',1);
        if (hyphen == std::string::npos)
        {
            if (!safeStrToUInt(s, t))
            {
                msg_error("set") << "parsing \""<<s<<"\": invalid unsigned integer";
                continue;
            }
            _set.insert(t);
        }
        else
        {
            unsigned int t1,t2;
            int tinc;
            std::string s1(s,0,hyphen);
            if (!safeStrToUInt(s1, t1))
            {
                msg_error("set") << "parsing \""<<s<<"\": invalid unsigned integer";
                continue;
            }
            const std::string::size_type hyphen2 = s.find_first_of('-',hyphen+2);
            if (hyphen2 == std::string::npos)
            {
                std::string s2(s,hyphen+1);
                if (!safeStrToUInt(s2, t2))
                {
                    msg_error("set") << "parsing \""<<s<<"\": invalid unsigned integer";
                    continue;
                }
                tinc = (t1<t2) ? 1 : -1;
            }
            else
            {
                std::string s2(s,hyphen+1,hyphen2-hyphen-1);
                std::string s3(s,hyphen2+1);
                if (!safeStrToUInt(s2, t2) || !safeStrToInt(s3, tinc))
                {
                    msg_error("set") << "parsing \""<<s<<"\": invalid integer";
                    continue;
                }
                if (tinc == 0)
                {
                    msg_error("set") << "parsing \""<<s<<"\": increment is 0";
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
                    _set.insert(t);
            else
                for (t=t1; t<=t2; t=(unsigned int)((int)t+tinc))
                    _set.insert(t);
        }
    }
    if( in.rdstate() & std::ios_base::eofbit ) { in.clear(); }
    return in;
}

} // namespace std

