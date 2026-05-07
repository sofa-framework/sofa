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

namespace sofa::geometry
{

/// The enumeration used to give unique identifiers to Topological objects.
enum class ElementType : sofa::Size
{
    UNKNOWN,
    POINT,
    EDGE,
    TRIANGLE,
    QUAD,
    TETRAHEDRON,
    HEXAHEDRON,
    PRISM,
    PYRAMID,
    SIZE
};

constexpr sofa::Size NumberOfElementType = static_cast<sofa::Size>(sofa::geometry::ElementType::SIZE);

constexpr const char* elementTypeToString(ElementType type)
{
    switch (type)
    {
    case ElementType::POINT: { return "Point"; }
    case ElementType::EDGE: { return "Edge"; }
    case ElementType::TRIANGLE: { return "Triangle"; }
    case ElementType::QUAD: { return "Quad"; }
    case ElementType::TETRAHEDRON: { return "Tetrahedron"; }
    case ElementType::HEXAHEDRON: { return "Hexahedron"; }
    case ElementType::PRISM: { return "Prism"; }
    case ElementType::PYRAMID: { return "Pyramid"; }
    default: 
        return "Unknown";
    }
}

} // namespace sofa::geometry
