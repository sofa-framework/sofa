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
#define SOFA_TOPOLOGY_TOPOLOGYELEMENTINFO_DEFINITION

#include <sofa/topology/ElementInfo.h>
#include <sofa/topology/ElementType.h>

namespace sofa::topology
{

template<>
SOFA_TOPOLOGY_API ElementType ElementInfo<Point>::type()
{
    return ElementType::POINT;
}

template<>
SOFA_TOPOLOGY_API const char* ElementInfo<Point>::name()
{
    return "Point";
}

template<>
SOFA_TOPOLOGY_API ElementType ElementInfo<Edge>::type()
{
    return ElementType::EDGE;
}

template<>
SOFA_TOPOLOGY_API const char* ElementInfo<Edge>::name()
{
    return "Edge";
}

template<>
SOFA_TOPOLOGY_API ElementType ElementInfo<Triangle>::type()
{
    return ElementType::TRIANGLE;
}

template<>
SOFA_TOPOLOGY_API const char* ElementInfo<Triangle>::name()
{
    return "Triangle";
}

template<>
SOFA_TOPOLOGY_API ElementType ElementInfo<Quad>::type()
{
    return ElementType::QUAD;
}

template<>
SOFA_TOPOLOGY_API const char* ElementInfo<Quad>::name()
{
    return "Quad";
}

template<>
SOFA_TOPOLOGY_API ElementType ElementInfo<Tetrahedron>::type()
{
    return ElementType::TETRAHEDRON;
}

template<>
SOFA_TOPOLOGY_API const char* ElementInfo<Tetrahedron>::name()
{
    return "Tetrahedron";
}

template<>
SOFA_TOPOLOGY_API ElementType ElementInfo<Pyramid>::type()
{
    return ElementType::PYRAMID;
}

template<>
SOFA_TOPOLOGY_API const char* ElementInfo<Pyramid>::name()
{
    return "Pyramid";
}

template<>
SOFA_TOPOLOGY_API ElementType ElementInfo<Pentahedron>::type()
{
    return ElementType::PENTAHEDRON;
}

template<>
SOFA_TOPOLOGY_API const char* ElementInfo<Pentahedron>::name()
{
    return "Pentahedron";
}

template<>
SOFA_TOPOLOGY_API ElementType ElementInfo<Hexahedron>::type()
{
    return ElementType::HEXAHEDRON;
}

template<>
SOFA_TOPOLOGY_API const char* ElementInfo<Hexahedron>::name()
{
    return "Hexahedron";
}

} // namespace sofa::topology
