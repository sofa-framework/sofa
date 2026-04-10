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

#include <iterator>

namespace sofa::helper
{

/**
 * Simple alternative to std::ranges::iota_view, compatible with pre-C++20
 */
template <typename T>
class IotaView
{
public:
    struct iota_iterator
    {
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;
        using reference = T&;
        using pointer = T*;

        iota_iterator() = default;
        explicit iota_iterator(T val) : val_(val) {}

        T operator*() const { return val_; }
        iota_iterator& operator++() { ++val_; return *this; }
        bool operator!=(const iota_iterator& other) const { return val_ != other.val_; }

    private:
        T val_{};
    };

    using iterator = iota_iterator;
    using value_type = T;

    IotaView(T start, T end) : m_start(start), m_end(end) {}

    iterator begin() const { return iterator(m_start); }
    iterator end() const { return iterator(m_end); }

    [[nodiscard]] T size() const { return m_end - m_start; }
    [[nodiscard]] T operator[](T i) const { return m_start + i; }
    T front() const { return m_start; }
    T back() const { return m_end - 1; }
    [[nodiscard]] bool empty() const { return m_start == m_end; }

private:
    T m_start;
    T m_end;
};


}
