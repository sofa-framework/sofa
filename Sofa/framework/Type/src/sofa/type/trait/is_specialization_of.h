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
#include <type_traits>

namespace sofa::type::trait
{
/**
 * @brief Trait to check if a type `T` is a specialization of a given template class `Template`.
 *
 * The `is_specialization_of` trait is a compile-time check to determine if a type `T`
 * is a specialization of a particular template class `Template`. This trait works with
 * template classes that accept any number of template parameters.
 *
 * Example usage:
 * @code
 * template <typename T1, typename T2>
 * class Foo {};
 *
 * template <typename T>
 * class Bar {};
 *
 * class Baz {};
 *
 * static_assert(is_specialization_of<Foo<int, double>, Foo>::value, "Foo<int, double> is a Foo!");
 * static_assert(is_specialization_of<Bar<int>, Bar>::value, "Bar<int> is a Bar!");
 * static_assert(!is_specialization_of<int, Foo>::value, "int is not a Foo specialization.");
 * static_assert(!is_specialization_of<Baz, Bar>::value, "Baz is not a Bar specialization.");
 * @endcode
 *
 * @tparam T The type to be checked. This is the type that you want to determine whether it is a specialization of `Template`.
 * @tparam Template The template class to check against. This can be any template class that accepts one or more template parameters.
 */
template <typename T, template <typename...> class Template>
struct is_specialization_of : std::false_type {};

/**
 * @brief Partial specialization for the case where `T` is an instance of `Template<Args...>`.
 *
 * This specialization checks if `T` matches the form `Template<Args...>`, meaning `T` is a specialization
 * of the template class `Template` with specific template parameters `Args...`.
 *
 * @tparam Template The template class that `T` is expected to be an instance of.
 * @tparam Args The actual template parameters used in the instantiation of `Template`.
 */
template <template <typename...> class Template, typename... Args>
struct is_specialization_of<Template<Args...>, Template> : std::true_type {};

/**
 * @brief Helper variable template to simplify the syntax for checking if `T` is a specialization of `Template`.
 *
 * This variable template provides a cleaner and more concise way to use the `is_specialization_of` trait.
 * Instead of writing `is_specialization_of<T, Template>::value`, you can use `is_specialization_of_v<T, Template>`.
 *
 * @tparam T The type to be checked.
 * @tparam Template The template class to check against.
 * @see is_specialization_of
 */
template <typename T, template <typename...> class Template>
inline constexpr bool is_specialization_of_v = is_specialization_of<T, Template>::value;

}
