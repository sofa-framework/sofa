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
 * 16 Mar 2017 - stop printing an extra space at end of <<.
 * 17 Jan 2017 - add std::enable_if to replace static_assert (Damien Marchal)
 * 29 Jun 2005 - remove boost includes and reverse iterators. (Jeremie Allard)
 * 23 Aug 2002 - fix for Non-MSVC compilers combined with MSVC libraries.
 * 05 Aug 2001 - minor update (Nico Josuttis)
 * 20 Jan 2001 - STLport fix (Beman Dawes)
 * 29 Sep 2000 - Initial Revision (Nico Josuttis)
 */

// See http://www.boost.org/libs/array for Documentation.

// FF added operator <
// JA added constructors from tuples
#pragma once

#include <sofa/type/config.h>
#include <sofa/type/fwd.h>

#include <cstddef>
#include <stdexcept>
#include <iterator>
#include <cassert>
#include <iostream>
#include <type_traits>
#include <algorithm>
#include <utility>


namespace sofa::type
{

template<class T, sofa::Size N>
class fixed_array
{
public:
    T elems[N]{};    // fixed-size array of elements of type T

    typedef T Array[N]; ///< name the array type

    // type definitions
    typedef T              value_type;
    typedef T*             iterator;
    typedef const T*       const_iterator;
    typedef T&             reference;
    typedef const T&       const_reference;
    typedef sofa::Size     size_type;
    typedef std::ptrdiff_t difference_type;

    constexpr fixed_array() {}

    /// Specific constructor for 1-element vectors.
    template<size_type NN = N, typename std::enable_if<NN == 1, int>::type = 0>
    explicit constexpr fixed_array(value_type r1) noexcept
    {
        elems[0] = r1;
    }

    template<typename... ArgsT,
        typename = std::enable_if_t< (std::is_convertible_v<ArgsT, value_type> && ...) >,
        typename = std::enable_if_t< (sizeof...(ArgsT) == N && sizeof...(ArgsT) > 1) >
    >
    constexpr fixed_array(ArgsT&&... r) noexcept
        : elems{static_cast<value_type>(std::forward< ArgsT >(r))...}
    {}

    // iterator support
    constexpr iterator begin() noexcept
    {
        return elems;
    }
    constexpr const_iterator begin() const noexcept
    {
        return elems;
    }
    constexpr const_iterator cbegin() const noexcept
    {
        return elems;
    }

    constexpr iterator end() noexcept
    {
        return elems+N;
    }
    constexpr const_iterator end() const noexcept
    {
        return elems+N;
    }
    constexpr const_iterator cend() const noexcept
    {
        return elems + N;
    }

    // operator[]
    constexpr reference operator[](size_type i)
    {
#ifndef NDEBUG
        assert(i<N && "index in fixed_array must be smaller than size");
#endif
        return elems[i];
    }
    constexpr const_reference operator[](size_type i) const
    {
#ifndef NDEBUG
        assert(i<N && "index in fixed_array must be smaller than size");
#endif
        return elems[i];
    }

    template< std::size_t I >
    [[nodiscard]] constexpr T& get() & noexcept
    {
        static_assert(I < N, "array index out of bounds");
        return elems[I];
    }

    template< std::size_t I >
    [[nodiscard]] constexpr const T& get() const& noexcept
    {
        static_assert(I < N, "array index out of bounds");
        return elems[I];
    }

    template< std::size_t I >
    [[nodiscard]] constexpr T&& get() && noexcept
    {
        static_assert(I < N, "array index out of bounds");
        return std::move(elems[I]);
    }

    template< std::size_t I >
    [[nodiscard]] constexpr const T&& get() const&& noexcept
    {
        static_assert(I < N, "array index out of bounds");
        return std::move(elems[I]);
    }


    // at() with range check
    constexpr reference at(size_type i)
    {
        rangecheck(i);
        return elems[i];
    }
    constexpr const_reference at(size_type i) const
    {
        rangecheck(i);
        return elems[i];
    }

    // front() and back()
    constexpr reference front()
    {
        return elems[0];
    }
    constexpr const_reference front() const
    {
        return elems[0];
    }
    constexpr reference back()
    {
        return elems[N-1];
    }
    constexpr const_reference back() const
    {
        return elems[N-1];
    }

    // size is constant
    static constexpr size_type size() noexcept
    {
        return N;
    }
    static bool empty() noexcept
    {
        return false;
    }
    static constexpr size_type max_size() noexcept
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
    constexpr const T* data() const noexcept
    {
        return elems;
    }

    /// direct access to array
    constexpr const Array& array() const noexcept
    {
        return elems;
    }

    /// direct access to array
    constexpr Array& array() noexcept
    {
        return elems;
    }

    // assignment with type conversion
    template <typename T2>
    constexpr fixed_array<T,N>& operator= (const fixed_array<T2,N>& rhs) noexcept
    {
        for (size_type i=0; i<N; i++)
            elems[i] = rhs[i];
        return *this;
    }

    // assign one value to all elements
    constexpr inline void assign (const T& value) noexcept
    {
        for (size_type i=0; i<N; i++)
            elems[i] = value;
    }

    inline friend std::ostream& operator << (std::ostream& out, const fixed_array<T,N>& a)
    {
        static_assert(N>0, "Cannot create a zero size arrays") ;
        for( size_type i=0; i<N-1; i++ )
            out << a.elems[i]<<" ";
        out << a.elems[N-1];
        return out;
    }

    inline friend std::istream& operator >> (std::istream& in, fixed_array<T,N>& a)
    {
        for( size_type i=0; i<N; i++ )
            in>>a.elems[i];
        return in;
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

template<typename... Ts>
constexpr auto make_array(Ts&&... ts) -> fixed_array<std::common_type_t<Ts...>, sizeof...(Ts)>
{
    return { std::forward<Ts>(ts)... };
}

/// Builds a fixed_array in which all elements have the same value
template<typename T, sofa::Size N>
constexpr sofa::type::fixed_array<T, N> makeHomogeneousArray(const T& value)
{
    sofa::type::fixed_array<T, N> container{};
    container.assign(value);
    return container;
}

/// Builds a fixed_array in which all elements have the same value
template<typename FixedArray>
constexpr FixedArray makeHomogeneousArray(const typename FixedArray::value_type& value)
{
    FixedArray container{};
    container.assign(value);
    return container;
}

/// Checks if v1 is lexicographically less than v2. Similar to std::lexicographical_compare
template<typename T, sofa::Size N>
constexpr bool
operator<(const fixed_array<T, N>& v1, const fixed_array<T, N>& v2 ) noexcept
{
    for( sofa::Size i=0; i<N; i++ )
    {
        if( v1[i] < v2[i] )
            return true;
        if( v2[i] < v1[i] )
            return false;
    }
    return false;
}

template<typename T, sofa::Size N>
constexpr bool
operator>(const fixed_array<T, N>& v1, const fixed_array<T, N>& v2 ) noexcept
{
    return v2 < v1;
}

#ifndef FIXED_ARRAY_CPP
extern template class SOFA_TYPE_API fixed_array<float, 2>;
extern template class SOFA_TYPE_API fixed_array<double, 2>;

extern template class SOFA_TYPE_API fixed_array<float, 3>;
extern template class SOFA_TYPE_API fixed_array<double, 3>;

extern template class SOFA_TYPE_API fixed_array<float, 4>;
extern template class SOFA_TYPE_API fixed_array<double, 4>;

extern template class SOFA_TYPE_API fixed_array<float, 5>;
extern template class SOFA_TYPE_API fixed_array<double, 5>;

extern template class SOFA_TYPE_API fixed_array<float, 6>;
extern template class SOFA_TYPE_API fixed_array<double, 6>;

extern template class SOFA_TYPE_API fixed_array<float, 7>;
extern template class SOFA_TYPE_API fixed_array<double, 7>;
#endif //FIXED_ARRAY_CPP

} // namespace sofa::type

namespace std
{
template<typename T, sofa::Size N>
struct tuple_size<::sofa::type::fixed_array<T, N> > : integral_constant<size_t, N> {};

template<std::size_t I, typename T, sofa::Size N>
struct tuple_element<I, ::sofa::type::fixed_array<T, N> >
{
    using type = T;
};
}
