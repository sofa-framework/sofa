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

#include <sofa/topology/config.h>

#include <sofa/type/fixed_array.h>
#include <sofa/geometry/ElementType.h>

#include <type_traits>

namespace sofa::topology
{

template <typename GeometryElement>
struct Element
{
    static constexpr auto NumberOfNodes = GeometryElement::NumberOfNodes;
    static constexpr sofa::geometry::ElementType Element_type = GeometryElement::Element_type;

    using ArrayType = sofa::type::fixed_array<sofa::Index, GeometryElement::NumberOfNodes>;
    using Size = sofa::Size;
    using value_type = sofa::Index;
    using iterator = typename ArrayType::iterator;
    using const_iterator = typename ArrayType::const_iterator;
    using reference = typename ArrayType::reference;
    using const_reference = typename ArrayType::const_reference;
    using size_type = sofa::Size;
    using difference_type = std::ptrdiff_t;

    static constexpr sofa::Size static_size = GeometryElement::NumberOfNodes;
    static constexpr sofa::Size size() { return static_size; }

    constexpr Element() noexcept
    {
        elems.fill(sofa::InvalidID);
    }

    template< typename... ArgsT>
    constexpr Element(ArgsT&&... args) noexcept
    requires((std::is_convertible_v<ArgsT, sofa::Index> && ...))
    : elems
    {{ static_cast<sofa::Index>(std::forward< ArgsT >(args))... }}
    {
        static_assert(GeometryElement::NumberOfNodes == sizeof...(ArgsT), "Trying to construct the element with an incorrect number of nodes.");
    }

    constexpr reference operator[](size_type i)
    {
        return elems[i];
    }
    constexpr const_reference operator[](size_type i) const
    {
        return elems[i];
    }

    constexpr reference at(size_type i)
    {
        return elems.at(i);
    }

    constexpr const_reference at(size_type i) const
    {
        return elems.at(i);
    }

    template< std::size_t I >
    [[nodiscard]] constexpr reference get() & noexcept requires( I < static_size )
    {
        return elems[I];
    }

    template< std::size_t I >
    [[nodiscard]] constexpr const_reference get() const& noexcept requires( I < static_size )
    {
        return elems[I];
    }

    template< std::size_t I >
    [[nodiscard]] constexpr value_type&& get() && noexcept requires( I < static_size )
    {
        return std::move(elems[I]);
    }

    template< std::size_t I >
    [[nodiscard]] constexpr const value_type&& get() const&& noexcept requires( I < static_size )
    {
        return std::move(elems[I]);
    }

    constexpr iterator begin() noexcept
    {
        return elems.begin();
    }
    constexpr const_iterator begin() const noexcept
    {
        return elems.begin();
    }
    constexpr const_iterator cbegin() const noexcept
    {
        return elems.cbegin();
    }

    constexpr iterator end() noexcept
    {
        return elems.end();
    }
    constexpr const_iterator end() const noexcept
    {
        return elems.end();
    }
    constexpr const_iterator cend() const noexcept
    {
        return elems.cend();
    }

    bool operator<(const Element& other) const
    {
        return elems < other.elems;
    }

    const ArrayType& array() const
    {
        return elems;
    }

    friend std::ostream& operator<<(std::ostream& out, const Element<GeometryElement>& a)
    {
        return sofa::type::extraction(out, a.elems);
    }
    friend std::istream& operator>>(std::istream& in, Element<GeometryElement>& a)
    {
        return sofa::type::insertion(in, a.elems);
    }

private:
    ArrayType elems{};
};

} // namespace sofa::topology
