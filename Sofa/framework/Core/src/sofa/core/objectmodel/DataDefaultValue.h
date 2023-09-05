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
#include <optional>


namespace sofa::core::objectmodel
{

/**
 * Variable template used to know if a Data must store its default value, based
 * on its underlying type. By default, all types don't store their default
 * values. This variable template must be specialized in order to save its
 * default value (see other specializations of this variable template as an
 * example).
 */
template<class T, typename = void>
inline constexpr bool must_store_data_default_value = false;


/**
 * Used as a base class of Data<T>. It is either empty, depending on the type of
 * T, or it contains a value of type T used to store the Data default value.
 * If it is empty, empty base optimization applies (https://en.cppreference.com/w/cpp/language/ebo)
 */
template<class, typename = void>
struct DataDefaultValue
{
    static constexpr bool storeDefaultValue = false;
};

template<class T>
struct DataDefaultValue<T, std::enable_if_t<must_store_data_default_value<T>>>
{
    static constexpr bool storeDefaultValue = true;
    T m_defaultValue;
};


/**
 * Data<T> stores its default value if T is scalar type
 */
template<class T>
inline constexpr bool must_store_data_default_value<T, std::enable_if_t<std::is_scalar_v<T>>> = true;

/**
 * Data<T> stores its default value if T is a fixed_array
 */
template<class T, sofa::Size N>
inline constexpr bool must_store_data_default_value<sofa::type::fixed_array<T, N>> = true;


}
