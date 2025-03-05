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

#include <sofa/geometry/config.h>
#include <sofa/geometry/ElementType.h>
#include <sofa/geometry/Triangle.h>

namespace sofa::geometry
{

struct Quad
{
    static constexpr sofa::Size NumberOfNodes = 4;
    static constexpr ElementType Element_type = ElementType::QUAD;

    Quad() = delete;

    /**
    * @brief	Compute the area of a quadrilateral
    * @remark	The order of nodes needs to be consecutive
    * @tparam   Node iterable container
    * @tparam   T scalar
    * @param	n0,n1,n2,n3 nodes of the quadrilateral
    * @return	Area of the quadrilateral (a T scalar)
    */
    template<typename Node,
             typename T = std::decay_t<decltype(*std::begin(std::declval<Node>()))>,
             typename = std::enable_if_t<std::is_scalar_v<T>>
    > 
    [[nodiscard]]
    static constexpr auto area(const Node& n0, const Node& n1, const Node& n2, const Node& n3)
    {

        return Triangle::area(n0, n1, n2) + Triangle::area(n0, n2, n3);
    }
};

} // namespace sofa::geometry
