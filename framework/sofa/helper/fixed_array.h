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
/* The following code declares class array,
 * an STL container (as wrapper) for arrays of constant size.
 *
 * See
 *      http://www.josuttis.com/cppcode
 * for details and the latest version.
 *
 * (C) Copyright Nicolai M. Josuttis 2001.
 * Permission to copy, use, modify, sell and distribute this software
 * is granted provided this copyright notice appears in all copies.
 * This software is provided "as is" without express or implied
 * warranty, and with no claim as to its suitability for any purpose.
 *
 * 29 Jun 2005 - remove boost includes and reverse iterators. (Jeremie Allard)
 * 23 Aug 2002 - fix for Non-MSVC compilers combined with MSVC libraries.
 * 05 Aug 2001 - minor update (Nico Josuttis)
 * 20 Jan 2001 - STLport fix (Beman Dawes)
 * 29 Sep 2000 - Initial Revision (Nico Josuttis)
 */

// See http://www.boost.org/libs/array for Documentation.

// FF added operator <
// JA added constructors from tuples
#ifndef SOFA_HELPER_FIXED_ARRAY_H
#define SOFA_HELPER_FIXED_ARRAY_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/helper/system/config.h>
#include <sofa/SofaFramework.h>
#include <boost/static_assert.hpp>

#include <cstddef>
#include <stdexcept>
#include <iterator>
#include <algorithm>
#include <math.h>
#include <cassert>


namespace sofa
{

namespace helper
{

template<class T, std::size_t N>
class fixed_array
{
public:
    T elems[N];    // fixed-size array of elements of type T

    typedef T Array[N]; ///< name the array type

public:
    // type definitions
    typedef T              value_type;
    typedef T*             iterator;
    typedef const T*       const_iterator;
    typedef T&             reference;
    typedef const T&       const_reference;
    typedef std::size_t    size_type;
    typedef std::ptrdiff_t difference_type;

    fixed_array()
    {
    }


    /// Specific constructor for 1-element vectors.
    explicit fixed_array(value_type r1)
    {
        BOOST_STATIC_ASSERT(N==1);
        this->elems[0]=r1;
    }

    /// Specific constructor for 2-elements vectors.
    fixed_array(value_type r1, value_type r2)
    {
        BOOST_STATIC_ASSERT(N == 2);
        this->elems[0]=r1;
        this->elems[1]=r2;
    }

    /// Specific constructor for 3-elements vectors.
    fixed_array(value_type r1, value_type r2, value_type r3)
    {
        BOOST_STATIC_ASSERT(N == 3);
        this->elems[0]=r1;
        this->elems[1]=r2;
        this->elems[2]=r3;
    }

    /// Specific constructor for 4-elements vectors.
    fixed_array(value_type r1, value_type r2, value_type r3, value_type r4)
    {
        BOOST_STATIC_ASSERT(N == 4);
        this->elems[0]=r1;
        this->elems[1]=r2;
        this->elems[2]=r3;
        this->elems[3]=r4;
    }

    /// Specific constructor for 5-elements vectors.
    fixed_array(value_type r1, value_type r2, value_type r3, value_type r4, value_type r5)
    {
        BOOST_STATIC_ASSERT(N == 5);
        this->elems[0]=r1;
        this->elems[1]=r2;
        this->elems[2]=r3;
        this->elems[3]=r4;
        this->elems[4]=r5;
    }

    /// Specific constructor for 6-elements vectors.
    fixed_array(value_type r1, value_type r2, value_type r3, value_type r4, value_type r5, value_type r6)
    {
        BOOST_STATIC_ASSERT(N == 6);
        this->elems[0]=r1;
        this->elems[1]=r2;
        this->elems[2]=r3;
        this->elems[3]=r4;
        this->elems[4]=r5;
        this->elems[5]=r6;
    }

    /// Specific constructor for 7-elements vectors.
    fixed_array(value_type r1, value_type r2, value_type r3, value_type r4, value_type r5, value_type r6, value_type r7)
    {
        BOOST_STATIC_ASSERT(N == 7);
        this->elems[0]=r1;
        this->elems[1]=r2;
        this->elems[2]=r3;
        this->elems[3]=r4;
        this->elems[4]=r5;
        this->elems[5]=r6;
        this->elems[6]=r7;
    }

    /// Specific constructor for 8-elements vectors.
    fixed_array(value_type r1, value_type r2, value_type r3, value_type r4, value_type r5, value_type r6, value_type r7, value_type r8)
    {
        BOOST_STATIC_ASSERT(N == 8);
        this->elems[0]=r1;
        this->elems[1]=r2;
        this->elems[2]=r3;
        this->elems[3]=r4;
        this->elems[4]=r5;
        this->elems[5]=r6;
        this->elems[6]=r7;
        this->elems[7]=r8;
    }

    /// Specific constructor for 9-elements vectors.
    fixed_array(value_type r1, value_type r2, value_type r3, value_type r4, value_type r5, value_type r6, value_type r7, value_type r8, value_type r9)
    {
        BOOST_STATIC_ASSERT(N == 9);
        this->elems[0]=r1;
        this->elems[1]=r2;
        this->elems[2]=r3;
        this->elems[3]=r4;
        this->elems[4]=r5;
        this->elems[5]=r6;
        this->elems[6]=r7;
        this->elems[7]=r8;
        this->elems[8]=r9;
    }

    /// Specific constructor for 10-elements vectors.
    fixed_array(value_type r1, value_type r2, value_type r3, value_type r4, value_type r5, value_type r6, value_type r7, value_type r8, value_type r9, value_type r10)
    {
        BOOST_STATIC_ASSERT(N == 10);
        this->elems[0]=r1;
        this->elems[1]=r2;
        this->elems[2]=r3;
        this->elems[3]=r4;
        this->elems[4]=r5;
        this->elems[5]=r6;
        this->elems[6]=r7;
        this->elems[7]=r8;
        this->elems[8]=r9;
        this->elems[9]=r10;
    }


    // iterator support
    iterator begin()
    {
        return elems;
    }
    const_iterator begin() const
    {
        return elems;
    }
    iterator end()
    {
        return elems+N;
    }
    const_iterator end() const
    {
        return elems+N;
    }

    // operator[]
    reference operator[](size_type i)
    {
#ifndef NDEBUG
        assert(i<N && "index in fixed_array must be smaller than size");
#endif
        return elems[i];
    }
    const_reference operator[](size_type i) const
    {
#ifndef NDEBUG
        assert(i<N && "index in fixed_array must be smaller than size");
#endif
        return elems[i];
    }

    // at() with range check
    reference at(size_type i)
    {
        rangecheck(i);
        return elems[i];
    }
    const_reference at(size_type i) const
    {
        rangecheck(i);
        return elems[i];
    }

    // front() and back()
    reference front()
    {
        return elems[0];
    }
    const_reference front() const
    {
        return elems[0];
    }
    reference back()
    {
        return elems[N-1];
    }
    const_reference back() const
    {
        return elems[N-1];
    }

    // size is constant
    static size_type size()
    {
        return N;
    }
    static bool empty()
    {
        return false;
    }
    static size_type max_size()
    {
        return N;
    }
    enum { static_size = N };

    // swap (note: linear complexity)
    void swap (fixed_array<T,N>& y)
    {
        std::swap_ranges(begin(),end(),y.begin());
    }

    // direct access to data
    const T* data() const
    {
        return elems;
    }

    /// direct access to array
    const Array& array() const
    {
        return elems;
    }

    /// direct access to array
    Array& array()
    {
        return elems;
    }

    // assignment with type conversion
    template <typename T2>
    fixed_array<T,N>& operator= (const fixed_array<T2,N>& rhs)
    {
        //std::copy(rhs.begin(),rhs.end(), begin());
        for (size_type i=0; i<N; i++)
            elems[i] = rhs[i];
        return *this;
    }

    // assign one value to all elements
    void assign (const T& value)
    {
        //std::fill_n(begin(),size(),value);
        for (size_type i=0; i<N; i++)
            elems[i] = value;
    }

    inline friend std::ostream& operator << (std::ostream& out, const fixed_array<T,N>& a)
    {
        for( size_type i=0; i<N; i++ )
            out<<a.elems[i]<<" ";
        return out;
    }

    inline friend std::istream& operator >> (std::istream& in, fixed_array<T,N>& a)
    {
        for( size_type i=0; i<N; i++ )
            in>>a.elems[i];
        return in;
    }

    inline bool operator < (const fixed_array& v ) const
    {
        for( size_type i=0; i<N; i++ )
        {
            if( elems[i]<v[i] )
                return true;  // (*this)<v
            else if( elems[i]>v[i] )
                return false; // (*this)>v
        }
        return false; // (*this)==v
    }

private:

    // check range (may be private because it is static)
    static void rangecheck (size_type i)
    {
        if (i >= size())
        {
            throw std::range_error("fixed_array");
        }
    }

};

template<class T>
inline fixed_array<T, 2> make_array(const T& v0, const T& v1)
{
    fixed_array<T, 2> v;
    v[0] = v0;
    v[1] = v1;
    return v;
}

template<class T>
inline fixed_array<T, 3> make_array(const T& v0, const T& v1, const T& v2)
{
    fixed_array<T, 3> v;
    v[0] = v0;
    v[1] = v1;
    v[2] = v2;
    return v;
}

template<class T>
inline fixed_array<T, 4> make_array(const T& v0, const T& v1, const T& v2, const T& v3)
{
    fixed_array<T, 4> v;
    v[0] = v0;
    v[1] = v1;
    v[2] = v2;
    v[3] = v3;
    return v;
}

template<class T>
inline fixed_array<T, 5> make_array(const T& v0, const T& v1, const T& v2, const T& v3, const T& v4)
{
    fixed_array<T, 5> v;
    v[0] = v0;
    v[1] = v1;
    v[2] = v2;
    v[3] = v3;
    v[4] = v4;
    return v;
}

template<class T>
inline fixed_array<T, 6> make_array(const T& v0, const T& v1, const T& v2, const T& v3, const T& v4, const T& v5)
{
    fixed_array<T, 6> v;
    v[0] = v0;
    v[1] = v1;
    v[2] = v2;
    v[3] = v3;
    v[4] = v4;
    v[5] = v5;
    return v;
}

template<class T>
inline fixed_array<T, 7> make_array(const T& v0, const T& v1, const T& v2, const T& v3, const T& v4, const T& v5, const T& v6)
{
    fixed_array<T, 7> v;
    v[0] = v0;
    v[1] = v1;
    v[2] = v2;
    v[3] = v3;
    v[4] = v4;
    v[5] = v5;
    v[6] = v6;
    return v;
}

template<class T>
inline fixed_array<T, 8> make_array(const T& v0, const T& v1, const T& v2, const T& v3, const T& v4, const T& v5, const T& v6, const T& v7)
{
    fixed_array<T, 8> v;
    v[0] = v0;
    v[1] = v1;
    v[2] = v2;
    v[3] = v3;
    v[4] = v4;
    v[5] = v5;
    v[6] = v6;
    v[7] = v7;
    return v;
}

template<class T>
inline fixed_array<T, 9> make_array(const T& v0, const T& v1, const T& v2, const T& v3, const T& v4, const T& v5, const T& v6, const T& v7, const T& v8)
{
    fixed_array<T, 9> v;
    v[0] = v0;
    v[1] = v1;
    v[2] = v2;
    v[3] = v3;
    v[4] = v4;
    v[5] = v5;
    v[6] = v6;
    v[7] = v7;
    v[8] = v8;
    return v;
}

template<class T>
inline fixed_array<T, 10> make_array(const T& v0, const T& v1, const T& v2, const T& v3, const T& v4, const T& v5, const T& v6, const T& v7, const T& v8, const T& v9)
{
    fixed_array<T, 10> v;
    v[0] = v0;
    v[1] = v1;
    v[2] = v2;
    v[3] = v3;
    v[4] = v4;
    v[5] = v5;
    v[6] = v6;
    v[7] = v7;
    v[8] = v8;
    v[9] = v9;
    return v;
}

} // namespace helper

} // namespace sofa

#endif
