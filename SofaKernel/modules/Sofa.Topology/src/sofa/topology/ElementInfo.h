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

#include <sofa/topology/ElementType.h>
#include <sofa/topology/Point.h>
#include <sofa/topology/Edge.h>
#include <sofa/topology/Triangle.h>
#include <sofa/topology/Quad.h>
#include <sofa/topology/Pentahedron.h>
#include <sofa/topology/Tetrahedron.h>
#include <sofa/topology/Pyramid.h>
#include <sofa/topology/Hexahedron.h>

namespace sofa::topology
{

template<typename Element>
struct ElementInfo
{
    static ElementType type()
    {
        return ElementType();
    }

    static const char* name()
    {
        return "";
    }
};

#ifndef SOFA_TOPOLOGY_TOPOLOGYELEMENTINFO_DEFINITION
extern template struct SOFA_TOPOLOGY_API ElementInfo<Point>;
extern template struct SOFA_TOPOLOGY_API ElementInfo<Edge>;
extern template struct SOFA_TOPOLOGY_API ElementInfo<Triangle>;
extern template struct SOFA_TOPOLOGY_API ElementInfo<Quad>;
extern template struct SOFA_TOPOLOGY_API ElementInfo<Pentahedron>;
extern template struct SOFA_TOPOLOGY_API ElementInfo<Tetrahedron>;
extern template struct SOFA_TOPOLOGY_API ElementInfo<Pyramid>;
extern template struct SOFA_TOPOLOGY_API ElementInfo<Hexahedron>;
#endif

} // namespace sofa::topology
