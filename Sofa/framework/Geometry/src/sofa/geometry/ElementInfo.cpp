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
#define SOFA_GEOMETRY_ELEMENTINFO_DEFINITION

#include <sofa/geometry/ElementInfo.h>

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

template<>
SOFA_GEOMETRY_API ElementType ElementInfo<Point>::type()
{
    return ElementType::POINT;
}

template<>
SOFA_GEOMETRY_API const char* ElementInfo<Point>::name()
{
    return "Point";
}

template<>
SOFA_GEOMETRY_API ElementType ElementInfo<Edge>::type()
{
    return ElementType::EDGE;
}

template<>
SOFA_GEOMETRY_API const char* ElementInfo<Edge>::name()
{
    return "Edge";
}

template<>
SOFA_GEOMETRY_API ElementType ElementInfo<Triangle>::type()
{
    return ElementType::TRIANGLE;
}

template<>
SOFA_GEOMETRY_API const char* ElementInfo<Triangle>::name()
{
    return "Triangle";
}

template<>
SOFA_GEOMETRY_API ElementType ElementInfo<Quad>::type()
{
    return ElementType::QUAD;
}

template<>
SOFA_GEOMETRY_API const char* ElementInfo<Quad>::name()
{
    return "Quad";
}

template<>
SOFA_GEOMETRY_API ElementType ElementInfo<Tetrahedron>::type()
{
    return ElementType::TETRAHEDRON;
}

template<>
SOFA_GEOMETRY_API const char* ElementInfo<Tetrahedron>::name()
{
    return "Tetrahedron";
}

template<>
SOFA_GEOMETRY_API ElementType ElementInfo<Pyramid>::type()
{
    return ElementType::PYRAMID;
}

template<>
SOFA_GEOMETRY_API const char* ElementInfo<Pyramid>::name()
{
    return "Pyramid";
}

template<>
SOFA_GEOMETRY_API ElementType ElementInfo<Pentahedron>::type()
{
    return ElementType::PENTAHEDRON;
}

template<>
SOFA_GEOMETRY_API const char* ElementInfo<Pentahedron>::name()
{
    return "Pentahedron";
}

template<>
SOFA_GEOMETRY_API ElementType ElementInfo<Hexahedron>::type()
{
    return ElementType::HEXAHEDRON;
}

template<>
SOFA_GEOMETRY_API const char* ElementInfo<Hexahedron>::name()
{
    return "Hexahedron";
}

template struct SOFA_GEOMETRY_API ElementInfo<Edge>;
template struct SOFA_GEOMETRY_API ElementInfo<Hexahedron>;
template struct SOFA_GEOMETRY_API ElementInfo<Pentahedron>;
template struct SOFA_GEOMETRY_API ElementInfo<Point>;
template struct SOFA_GEOMETRY_API ElementInfo<Pyramid>;
template struct SOFA_GEOMETRY_API ElementInfo<Quad>;
template struct SOFA_GEOMETRY_API ElementInfo<Tetrahedron>;
template struct SOFA_GEOMETRY_API ElementInfo<Triangle>;

} // namespace sofa::geometry
