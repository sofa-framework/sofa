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
#include <sofa/defaulttype/config.h>

#include <iterator>

namespace sofa::defaulttype
{

/**
 * Iterator for RigidCoord and RigidDeriv types
 */
template<class MyRigid>
class RigidConstIterator
{
public:
    using iterator_category = std::forward_iterator_tag;
    using difference_type   = std::ptrdiff_t;
    using value_type        = typename MyRigid::Real;
    using pointer           = const value_type*;
    using reference         = const value_type&;

    using Pos = typename MyRigid::Pos;
    using Rot = typename MyRigid::Rot;

    static constexpr const value_type* orientation_ptr(const Rot& orientation)
    {
        if constexpr (std::is_scalar_v<Rot>)
            return &orientation;
        else
            return orientation.ptr();
    }

    constexpr RigidConstIterator(const Pos& center, const Rot& orientation, const sofa::Size i = 0)
        : m_ptr( i < Pos::total_size ? center.data() : orientation_ptr(orientation) + (i - Pos::total_size))
        , m_i(i)
        , m_center(center)
        , m_orientation(orientation)
    {}

    constexpr reference operator*() const { return *m_ptr; }
    constexpr pointer operator->() { return m_ptr; }

    // Prefix increment
    constexpr RigidConstIterator& operator++()
    {
        ++m_i;
        if (m_i == Pos::total_size)
        {
            m_ptr = orientation_ptr(m_orientation);
        }
        else
        {
            ++m_ptr;
        }
        return *this;
    }

    // Postfix increment
    constexpr RigidConstIterator operator++(int)
    {
        RigidConstIterator tmp = *this;
        ++(*this);
        return tmp;
    }

    friend constexpr bool operator== (const RigidConstIterator& a, const RigidConstIterator& b) { return a.m_ptr == b.m_ptr; }
    friend constexpr bool operator!= (const RigidConstIterator& a, const RigidConstIterator& b) { return a.m_ptr != b.m_ptr; }

private:
    pointer m_ptr { nullptr };
    sofa::Size m_i { 0 };

    const Pos& m_center;
    const Rot& m_orientation;
};

template<class MyRigid>
class RigidIterator : public RigidConstIterator<MyRigid>
{
public:
    using iterator_category = std::forward_iterator_tag;
    using difference_type   = std::ptrdiff_t;
    using value_type        = typename MyRigid::Real;
    using pointer           = value_type*;
    using reference         = value_type&;

    using Pos = typename MyRigid::Pos;
    using Rot = typename MyRigid::Rot;

    using Inherit = RigidConstIterator<MyRigid>;
    using Inherit::Inherit;

    constexpr reference operator*() const { return const_cast<reference>(Inherit::operator*()); }
    constexpr pointer operator->() { return this->m_ptr; }

    constexpr RigidIterator& operator++()
    {
        Inherit::operator++();
        return *this;
    }

    constexpr RigidIterator& operator++(int)
    {
        RigidIterator tmp = *this;
        Inherit::operator++();
        return tmp;
    }
};

}
