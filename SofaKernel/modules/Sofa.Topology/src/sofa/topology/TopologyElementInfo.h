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

#include <sofa/topology/TopologyElementType.h>
#include <sofa/topology/geometry/Point.h>
#include <sofa/topology/geometry/Edge.h>
#include <sofa/topology/geometry/Triangle.h>
#include <sofa/topology/geometry/Quad.h>
#include <sofa/topology/geometry/Pentahedron.h>
#include <sofa/topology/geometry/Tetrahedron.h>
#include <sofa/topology/geometry/Pyramid.h>
#include <sofa/topology/geometry/Hexahedron.h>

namespace sofa::topology
{

template<typename TopologyElement>
struct TopologyElementInfo
{
    static TopologyElementType type()
    {
        return TopologyElementType();
    }

    static const char* name()
    {
        return "";
    }
};

#ifndef SOFA_TOPOLOGY_TOPOLOGYELEMENTINFO_DEFINITION
extern template struct SOFA_TOPOLOGY_API TopologyElementInfo<geometry::Point>;
extern template struct SOFA_TOPOLOGY_API TopologyElementInfo<geometry::Edge>;
extern template struct SOFA_TOPOLOGY_API TopologyElementInfo<geometry::Triangle>;
extern template struct SOFA_TOPOLOGY_API TopologyElementInfo<geometry::Quad>;
extern template struct SOFA_TOPOLOGY_API TopologyElementInfo<geometry::Pentahedron>;
extern template struct SOFA_TOPOLOGY_API TopologyElementInfo<geometry::Tetrahedron>;
extern template struct SOFA_TOPOLOGY_API TopologyElementInfo<geometry::Pyramid>;
extern template struct SOFA_TOPOLOGY_API TopologyElementInfo<geometry::Hexahedron>;
#endif

} // namespace sofa::topology
