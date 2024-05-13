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

#include <sofa/geometry/ElementType.h>

#include <sofa/geometry/config.h>

#include <sofa/geometry/Edge.h>
#include <sofa/geometry/Hexahedron.h>
#include <sofa/geometry/Pentahedron.h>
#include <sofa/geometry/Point.h>
#include <sofa/geometry/Pyramid.h>
#include <sofa/geometry/Quad.h>
#include <sofa/geometry/Tetrahedron.h>
#include <sofa/geometry/Triangle.h>

namespace sofa::geometry
{

template<typename GeometryElement>
struct ElementInfo
{
    static ElementType type();

    static const char* name();
};


template <typename GeometryElement>
ElementType ElementInfo<GeometryElement>::type()
{
    return ElementType();
}

template <typename GeometryElement>
const char* ElementInfo<GeometryElement>::name()
{
    return "";
}

template<>
inline ElementType ElementInfo<Point>::type()
{
    return ElementType::POINT;
}

template<>
inline const char* ElementInfo<Point>::name()
{
    return "Point";
}

template<>
inline ElementType ElementInfo<Edge>::type()
{
    return ElementType::EDGE;
}

template<>
inline const char* ElementInfo<Edge>::name()
{
    return "Edge";
}

template<>
inline ElementType ElementInfo<Triangle>::type()
{
    return ElementType::TRIANGLE;
}

template<>
inline const char* ElementInfo<Triangle>::name()
{
    return "Triangle";
}

template<>
inline ElementType ElementInfo<Quad>::type()
{
    return ElementType::QUAD;
}

template<>
inline const char* ElementInfo<Quad>::name()
{
    return "Quad";
}

template<>
inline ElementType ElementInfo<Tetrahedron>::type()
{
    return ElementType::TETRAHEDRON;
}

template<>
inline const char* ElementInfo<Tetrahedron>::name()
{
    return "Tetrahedron";
}

template<>
inline ElementType ElementInfo<Pyramid>::type()
{
    return ElementType::PYRAMID;
}

template<>
inline const char* ElementInfo<Pyramid>::name()
{
    return "Pyramid";
}

template<>
inline ElementType ElementInfo<Pentahedron>::type()
{
    return ElementType::PENTAHEDRON;
}

template<>
inline const char* ElementInfo<Pentahedron>::name()
{
    return "Pentahedron";
}

template<>
inline ElementType ElementInfo<Hexahedron>::type()
{
    return ElementType::HEXAHEDRON;
}

template<>
inline const char* ElementInfo<Hexahedron>::name()
{
    return "Hexahedron";
}

#if !defined(SOFA_GEOMETRY_ELEMENTINFO_DEFINITION)
extern template struct SOFA_GEOMETRY_API ElementInfo<::sofa::geometry::Edge>;
extern template struct SOFA_GEOMETRY_API ElementInfo<::sofa::geometry::Hexahedron>;
extern template struct SOFA_GEOMETRY_API ElementInfo<::sofa::geometry::Pentahedron>;
extern template struct SOFA_GEOMETRY_API ElementInfo<::sofa::geometry::Point>;
extern template struct SOFA_GEOMETRY_API ElementInfo<::sofa::geometry::Pyramid>;
extern template struct SOFA_GEOMETRY_API ElementInfo<::sofa::geometry::Quad>;
extern template struct SOFA_GEOMETRY_API ElementInfo<::sofa::geometry::Tetrahedron>;
extern template struct SOFA_GEOMETRY_API ElementInfo<::sofa::geometry::Triangle>;
#endif

} // namespace sofa::geometry
