/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_HELPER_deque_H
#define SOFA_HELPER_deque_H

#include <deque>
#include <string>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <stdlib.h>

#include <sofa/SofaFramework.h>

namespace sofa
{

namespace helper
{

//======================================================================
///	Same as std::deque, + input/output operators
///
///	\author Christophe Guebert, 2013
///
//======================================================================
template<class T,
      class Alloc = std::allocator<T> >
class deque: public std::deque<T,Alloc>
{
public:

    /// size_type
    typedef typename std::deque<T,Alloc>::size_type size_type;
    /// reference to a value (read-write)
    typedef typename std::deque<T,Alloc>::reference reference;
    /// const reference to a value (read only)
    typedef typename std::deque<T,Alloc>::const_reference const_reference;
    /// iterator
    typedef typename std::deque<T,Alloc>::iterator iterator;
    /// const iterator
    typedef typename std::deque<T,Alloc>::const_iterator const_iterator;

    /// Basic constructor
    deque() {}
    /// Constructor
    deque(const std::deque<T, Alloc>& x): std::deque<T,Alloc>(x) {}
    /// Constructor
    deque<T, Alloc>& operator=(const std::deque<T, Alloc>& x)
    {
        std::deque<T,Alloc>::operator = (x);
        return (*this);
    }

#ifdef __STL_MEMBER_TEMPLATES
    /// Constructor
    template <class InputIterator>
    deque(InputIterator first, InputIterator last): std::deque<T,Alloc>(first,last) {}
#else /* __STL_MEMBER_TEMPLATES */
    /// Constructor
    deque(const_iterator first, const_iterator last): std::deque<T,Alloc>(first,last) {}
#endif /* __STL_MEMBER_TEMPLATES */

    std::ostream& write(std::ostream& os) const
    {
        if( this->size()>0 )
        {
            for( unsigned int i=0; i<this->size()-1; ++i ) os<<(*this)[i]<<" ";
            os<<(*this)[this->size()-1];
        }
        return os;
    }

    std::istream& read(std::istream& in)
    {
        T t=T();
        this->clear();
        while(in>>t)
            this->push_back(t);
        if( in.rdstate() & std::ios_base::eofbit ) { in.clear(); }
        return in;
    }

    /// Output stream
    inline friend std::ostream& operator<< ( std::ostream& os, const deque<T,Alloc>& vec )
    {
        return vec.write(os);
    }

    /// Input stream
    inline friend std::istream& operator>> ( std::istream& in, deque<T,Alloc>& vec )
    {
        return vec.read(in);
    }
};


/// Input stream
/// Specialization for reading deques of int and unsigned int using "A-B" notation for all integers between A and B, optionnally specifying a step using "A-B-step" notation.
template<>
inline std::istream& deque<int >::read( std::istream& in )
{
    int t;
    this->clear();
    std::string s;
    while(in>>s)
    {
        std::string::size_type hyphen = s.find_first_of('-',1);
        if (hyphen == std::string::npos)
        {
            t = atoi(s.c_str());
            this->push_back(t);
        }
        else
        {
            int t1,t2,tinc;
            std::string s1(s,0,hyphen);
            t1 = atoi(s1.c_str());
            std::string::size_type hyphen2 = s.find_first_of('-',hyphen+2);
            if (hyphen2 == std::string::npos)
            {
                std::string s2(s,hyphen+1);
                t2 = atoi(s2.c_str());
                tinc = (t1<t2) ? 1 : -1;
            }
            else
            {
                std::string s2(s,hyphen+1,hyphen2);
                std::string s3(s,hyphen2+1);
                t2 = atoi(s2.c_str());
                tinc = atoi(s3.c_str());
                if (tinc == 0)
                {
                    std::cerr << "ERROR parsing \""<<s<<"\": increment is 0\n";
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
                    this->push_back(t);
            else
                for (t=t1; t<=t2; t+=tinc)
                    this->push_back(t);
        }
    }
    if( in.rdstate() & std::ios_base::eofbit ) { in.clear(); }
    return in;
}

/// Output stream
/// Specialization for writing deques of unsigned char
template<>
inline std::ostream& deque<unsigned char >::write(std::ostream& os) const
{
    if( this->size()>0 )
    {
        for( unsigned int i=0; i<this->size()-1; ++i ) os<<(int)(*this)[i]<<" ";
        os<<(int)(*this)[this->size()-1];
    }
    return os;
}

/// Inpu stream
/// Specialization for writing deques of unsigned char
template<>
inline std::istream& deque<unsigned char >::read(std::istream& in)
{
    int t;
    this->clear();
    while(in>>t)
    {
        this->push_back((unsigned char)t);
    }
    if( in.rdstate() & std::ios_base::eofbit ) { in.clear(); }
    return in;
}

/// Input stream
/// Specialization for reading deques of int and unsigned int using "A-B" notation for all integers between A and B
template<>
inline std::istream& deque<unsigned int >::read( std::istream& in )
{
    unsigned int t;
    this->clear();
    std::string s;
    while(in>>s)
    {
        std::string::size_type hyphen = s.find_first_of('-',1);
        if (hyphen == std::string::npos)
        {
            t = atoi(s.c_str());
            this->push_back(t);
        }
        else
        {
            unsigned int t1,t2;
            int tinc;
            std::string s1(s,0,hyphen);
            t1 = (unsigned int)atoi(s1.c_str());
            std::string::size_type hyphen2 = s.find_first_of('-',hyphen+2);
            if (hyphen2 == std::string::npos)
            {
                std::string s2(s,hyphen+1);
                t2 = (unsigned int)atoi(s2.c_str());
                tinc = (t1<t2) ? 1 : -1;
            }
            else
            {
                std::string s2(s,hyphen+1,hyphen2);
                std::string s3(s,hyphen2+1);
                t2 = (unsigned int)atoi(s2.c_str());
                tinc = atoi(s3.c_str());
                if (tinc == 0)
                {
                    std::cerr << "ERROR parsing \""<<s<<"\": increment is 0\n";
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
                    this->push_back(t);
            else
                for (t=t1; t<=t2; t=(unsigned int)((int)t+tinc))
                    this->push_back(t);
        }
    }
    if( in.rdstate() & std::ios_base::eofbit ) { in.clear(); }
    return in;
}

} // namespace helper

} // namespace sofa

#endif
