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
#pragma once

#include <sofa/type/config.h>

#include <iostream>
#include <array>
#include <tuple>

namespace sofa::type
{

template<
    class T,
    std::size_t N
>
using fixed_array = std::array<T, N>;

namespace trait
{

// Trait to detect if T is a std::array
template<typename>
struct is_std_array : std::false_type {};

template<typename T, std::size_t N>
struct is_std_array<sofa::type::fixed_array<T, N>> : std::true_type {};

template<class T>
concept StdArray = is_std_array<T>::value;

template<class T>
concept FixedArrayLike = StdArray<T> || requires(std::remove_cv_t<T> t, const std::remove_cv_t<T> ct)
{
    T::static_size;

    {t.begin()} -> std::convertible_to<typename T::iterator>;
    {t.end()} -> std::convertible_to<typename T::iterator>;

    {ct.begin()} -> std::convertible_to<typename T::const_iterator>;
    {ct.end()} -> std::convertible_to<typename T::const_iterator>;

    { t[0] } -> std::convertible_to<typename T::value_type>;
};

template<class T>
static constexpr sofa::Size staticSize = 0;

template <typename T>
concept HasStaticSize = requires { T::static_size; };

template<HasStaticSize T>
static constexpr sofa::Size staticSize<T> = T::static_size;

template<sofa::type::trait::StdArray T>
static constexpr sofa::Size staticSize<T> = std::tuple_size_v<T>;


template<typename T>
concept Streamable = requires(std::ostream& os, const T& value)
{
    { os << value } -> std::same_as<std::ostream&>;
};

template<typename T>
concept InputStreamable = requires(std::istream& is, T& value)
{
    { is >> value } -> std::same_as<std::istream&>;
};
}

/// Builds a fixed_array in which all elements have the same value
template<typename T, size_t N>
constexpr sofa::type::fixed_array<T, N> makeHomogeneousArray(const T& value)
{
    sofa::type::fixed_array<T, N> container{};
    container.fill(value);
    return container;
}

/// Builds a fixed_array in which all elements have the same value
template<typename FixedArray>
constexpr FixedArray makeHomogeneousArray(const typename FixedArray::value_type& value)
{
    FixedArray container{};
    container.fill(value);
    return container;
}

template<typename... Ts>
constexpr auto make_array(Ts&&... ts) -> fixed_array<std::common_type_t<Ts...>, sizeof...(Ts)>
{
    return { std::forward<Ts>(ts)... };
}

template<sofa::type::trait::FixedArrayLike T>
requires sofa::type::trait::Streamable<typename T::value_type>
std::ostream& extraction(std::ostream& out, const T& a)
{
    for (std::size_t i = 0; i < trait::staticSize<T> - 1; i++)
    {
        out << a[i] << " ";
    }
    out << a[trait::staticSize<T> - 1];
    return out;
}

template<sofa::type::trait::FixedArrayLike T>
requires sofa::type::trait::InputStreamable<typename T::value_type>
std::istream& insertion(std::istream& in, T& a)
{
    for (auto& elem : a)
    {
        in >> elem;
    }
    return in;
}

}

namespace std
{
template<class T, std::size_t N>
requires sofa::type::trait::Streamable<T>
std::ostream& operator << (std::ostream& out, const std::array<T, N>& a)
{
    return sofa::type::extraction(out, a);
}

template<class T, std::size_t N>
requires sofa::type::trait::InputStreamable<T>
std::istream& operator >> (std::istream& in, std::array<T, N>& a)
{
    return sofa::type::insertion(in, a);
}

}
