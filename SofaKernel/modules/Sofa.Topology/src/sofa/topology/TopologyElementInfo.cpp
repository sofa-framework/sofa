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

#include <sofa/topology/TopologyElementInfo.h>
#include <sofa/topology/TopologyElementType.h>

namespace sofa::topology
{

template<>
SOFA_TOPOLOGY_API TopologyElementType TopologyElementInfo<geometry::Point>::type()
{
    return TopologyElementType::POINT;
}

template<>
SOFA_TOPOLOGY_API const char* TopologyElementInfo<geometry::Point>::name()
{
    return "Point";
}

template<>
SOFA_TOPOLOGY_API TopologyElementType TopologyElementInfo<geometry::Edge>::type()
{
    return TopologyElementType::EDGE;
}

template<>
SOFA_TOPOLOGY_API const char* TopologyElementInfo<geometry::Edge>::name()
{
    return "Edge";
}

template<>
SOFA_TOPOLOGY_API TopologyElementType TopologyElementInfo<geometry::Triangle>::type()
{
    return TopologyElementType::TRIANGLE;
}

template<>
SOFA_TOPOLOGY_API const char* TopologyElementInfo<geometry::Triangle>::name()
{
    return "Triangle";
}

template<>
SOFA_TOPOLOGY_API TopologyElementType TopologyElementInfo<geometry::Quad>::type()
{
    return TopologyElementType::QUAD;
}

template<>
SOFA_TOPOLOGY_API const char* TopologyElementInfo<geometry::Quad>::name()
{
    return "Quad";
}

template<>
SOFA_TOPOLOGY_API TopologyElementType TopologyElementInfo<geometry::Tetrahedron>::type()
{
    return TopologyElementType::TETRAHEDRON;
}

template<>
SOFA_TOPOLOGY_API const char* TopologyElementInfo<geometry::Tetrahedron>::name()
{
    return "Tetrahedron";
}

template<>
SOFA_TOPOLOGY_API TopologyElementType TopologyElementInfo<geometry::Pyramid>::type()
{
    return TopologyElementType::PYRAMID;
}

template<>
SOFA_TOPOLOGY_API const char* TopologyElementInfo<geometry::Pyramid>::name()
{
    return "Pyramid";
}

template<>
SOFA_TOPOLOGY_API TopologyElementType TopologyElementInfo<geometry::Pentahedron>::type()
{
    return TopologyElementType::PENTAHEDRON;
}

template<>
SOFA_TOPOLOGY_API const char* TopologyElementInfo<geometry::Pentahedron>::name()
{
    return "Pentahedron";
}

template<>
SOFA_TOPOLOGY_API TopologyElementType TopologyElementInfo<geometry::Hexahedron>::type()
{
    return TopologyElementType::HEXAHEDRON;
}

template<>
SOFA_TOPOLOGY_API const char* TopologyElementInfo<geometry::Hexahedron>::name()
{
    return "Hexahedron";
}

} // namespace sofa::topology
