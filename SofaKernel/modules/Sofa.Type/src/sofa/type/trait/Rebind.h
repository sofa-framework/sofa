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

namespace sofa::type
{
    // primary template handles types that have no nested ::rebind_to member:
    template< class T, class OtherType, class = void >
    struct HasRebindTypedef : std::false_type { };

    // specialization recognizes types that do have a nested ::rebind_to member:
    template< class T, class OtherType >
    struct HasRebindTypedef<T, OtherType, std::void_t<typename T::template rebind_to<OtherType> > > : std::true_type { };

    /**
     * Depending on the type _T, has a public member typedef to. Otherwise, there is no member typedef (this is the
     * case of this implementation).
     */
    template<class _T, class _OtherType, class Enable = void>
    struct Rebind {};

    /**
     * \brief Specialization for types that do have a nested ::rebind_to member. In this implementation, Rebind has
     * a public member typedef \ref to. It corresponds to the typedef ::rebind_to.
     *
     * \tparam _T Type that does have a nested ::rebind_to member
     */
    template<class _T, class _OtherType>
    struct Rebind<_T, _OtherType, std::enable_if_t<HasRebindTypedef<_T, _OtherType>::value > >
    {
        using to = typename _T::template rebind_to<_OtherType>;
    };

    template <class...>
    inline constexpr auto deny = false;

    /**
     * \brief Specialization for types that do NOT have a nested ::rebind_to member. In this implementation, Rebind has
     * no public member typedef \ref to. If this implementation is chosen by the compiler (the number of template
     * parameters is probably different from 1), a compilation error occurs.
     * \tparam _T Type that does NOT have a nested ::rebind_to member
     */
    template<class _T, class _OtherType>
    struct Rebind<_T, _OtherType, std::enable_if_t<!HasRebindTypedef<_T, _OtherType>::value > >
    {
        static_assert(deny<_T>, "_T must match _T<A>");
    };

    /**
     * \brief Specialization for types that do NOT have a nested ::rebind_to member. In this implementation, Rebind has
     * a public member typedef \ref to.
     * \tparam _T Type that does NOT have a nested ::rebind_to member
     */
    template<template<class> class _T, class A, class _OtherType>
    struct Rebind<_T<A>, _OtherType, std::enable_if_t<!HasRebindTypedef<_T<A>, _OtherType >::value > >
    {
        using to = _T<_OtherType>;
    };

    /**
     * Conveniant alias to ease usage of Rebind.
     *
     * Example:
     * 1) sofa::type::rebind_to< sofa::type::vector<int>, float> is of type sofa::type::vector<float>. In this example,
     * sofa::type::vector has a typedef rebind_to that will be used to deduce the type.
     * 2) sofa::type::rebind_to< sofa::type::Quat<float>, double> is of type sofa::type::Quat<double>. In this example,
     * sofa::type::Quat does not have a typdef rebind_to.
     * 3) It makes no sense to use sofa::type::rebind on types having more than one template parameter, such as
     * sofa::type::fixed_array. A compilation error would occur.
     */
    template <class T, class B>
    using rebind_to = typename Rebind<T, B>::to;
}
