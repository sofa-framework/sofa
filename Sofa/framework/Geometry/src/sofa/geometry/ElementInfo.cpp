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

template struct SOFA_GEOMETRY_API ElementInfo<sofa::geometry::Edge>;
template struct SOFA_GEOMETRY_API ElementInfo<sofa::geometry::Hexahedron>;
template struct SOFA_GEOMETRY_API ElementInfo<sofa::geometry::Pentahedron>;
template struct SOFA_GEOMETRY_API ElementInfo<sofa::geometry::Point>;
template struct SOFA_GEOMETRY_API ElementInfo<sofa::geometry::Pyramid>;
template struct SOFA_GEOMETRY_API ElementInfo<sofa::geometry::Quad>;
template struct SOFA_GEOMETRY_API ElementInfo<sofa::geometry::Tetrahedron>;
template struct SOFA_GEOMETRY_API ElementInfo<sofa::geometry::Triangle>;

} // namespace sofa::geometry
