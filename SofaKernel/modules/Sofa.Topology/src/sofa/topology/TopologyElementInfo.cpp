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
struct SOFA_TOPOLOGY_API TopologyElementInfo<geometry::Point>
{
    static TopologyElementType type() { return TopologyElementType::POINT; }
    static const char* name() { return "Point"; }
};

template<>
struct SOFA_TOPOLOGY_API TopologyElementInfo<geometry::Edge>
{
    static TopologyElementType type() { return TopologyElementType::EDGE; }
    static const char* name() { return "Edge"; }
};

template<>
struct SOFA_TOPOLOGY_API TopologyElementInfo<geometry::Triangle>
{
    static TopologyElementType type() { return TopologyElementType::TRIANGLE; }
    static const char* name() { return "Triangle"; }
};

template<>
struct SOFA_TOPOLOGY_API TopologyElementInfo<geometry::Quad>
{
    static TopologyElementType type() { return TopologyElementType::QUAD; }
    static const char* name() { return "Quad"; }
};

template<>
struct SOFA_TOPOLOGY_API TopologyElementInfo<geometry::Tetrahedron>
{
    static TopologyElementType type() { return TopologyElementType::TETRAHEDRON; }
    static const char* name() { return "Tetrahedron"; }
};

template<>
struct SOFA_TOPOLOGY_API TopologyElementInfo<geometry::Pyramid>
{
    static TopologyElementType type() { return TopologyElementType::PYRAMID; }
    static const char* name() { return "Pyramid"; }
};

template<>
struct SOFA_TOPOLOGY_API TopologyElementInfo<geometry::Pentahedron>
{
    static TopologyElementType type() { return TopologyElementType::PENTAHEDRON; }
    static const char* name() { return "Pentahedron"; }
};

template<>
struct SOFA_TOPOLOGY_API TopologyElementInfo<geometry::Hexahedron>
{
    static TopologyElementType type() { return TopologyElementType::HEXAHEDRON; }
    static const char* name() { return "Hexahedron"; }
};

} // namespace sofa::topology
