#ifndef SOFA_COMPONENTS_COMMON_FIXED_ARRAY_H
#define SOFA_COMPONENTS_COMMON_FIXED_ARRAY_H

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

#include <cstddef>
#include <stdexcept>
#include <iterator>
#include <algorithm>
#include <math.h>

namespace Sofa
{

namespace Components
{

namespace Common
{

template<class T, std::size_t N>
class fixed_array
{
public:
    T elems[N];    // fixed-size array of elements of type T

public:
    // type definitions
    typedef T              value_type;
    typedef T*             iterator;
    typedef const T*       const_iterator;
    typedef T&             reference;
    typedef const T&       const_reference;
    typedef std::size_t    size_type;
    typedef std::ptrdiff_t difference_type;

    // iterator support
    iterator begin() { return elems; }
    const_iterator begin() const { return elems; }
    iterator end() { return elems+N; }
    const_iterator end() const { return elems+N; }

    // operator[]
    reference operator[](size_type i) { return elems[i]; }
    const_reference operator[](size_type i) const { return elems[i]; }

    // at() with range check
    reference at(size_type i) { rangecheck(i); return elems[i]; }
    const_reference at(size_type i) const { rangecheck(i); return elems[i]; }

    // front() and back()
    reference front() { return elems[0]; }
    const_reference front() const { return elems[0]; }
    reference back() { return elems[N-1]; }
    const_reference back() const { return elems[N-1]; }

    // size is constant
    static size_type size() { return N; }
    static bool empty() { return false; }
    static size_type max_size() { return N; }
    enum { static_size = N };

    // swap (note: linear complexity)
    void swap (fixed_array<T,N>& y)
    {
        std::swap_ranges(begin(),end(),y.begin());
    }

    // direct access to data
    const T* data() const { return elems; }

    // assignment with type conversion
    template <typename T2>
    fixed_array<T,N>& operator= (const fixed_array<T2,N>& rhs)
    {
        std::copy(rhs.begin(),rhs.end(), begin());
        return *this;
    }

    // assign one value to all elements
    void assign (const T& value)
    {
        std::fill_n(begin(),size(),value);
    }

private:

    // check range (may be private because it is static)
    static void rangecheck (size_type i)
    {
        if (i >= size()) { throw std::range_error("fixed_array"); }
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

} // namespace Common

} // namespace Components

} // namespace Sofa

#endif
