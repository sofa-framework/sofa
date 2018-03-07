/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/helper/system/config.h>
#include <SofaTopologyMapping/initTopologyMapping.h>


namespace sofa
{

namespace component
{


void initTopologyMapping()
{
    static bool first = true;
    if (first)
    {
        first = false;
    }
}


SOFA_LINK_CLASS(Mesh2PointMechanicalMapping)
SOFA_LINK_CLASS(SimpleTesselatedTetraMechanicalMapping)
SOFA_LINK_CLASS(CenterPointTopologicalMapping)
SOFA_LINK_CLASS(Edge2QuadTopologicalMapping)
SOFA_LINK_CLASS(Hexa2QuadTopologicalMapping)
SOFA_LINK_CLASS(Hexa2TetraTopologicalMapping)
SOFA_LINK_CLASS(Mesh2PointTopologicalMapping)
SOFA_LINK_CLASS(Quad2TriangleTopologicalMapping)
SOFA_LINK_CLASS(SimpleTesselatedHexaTopologicalMapping)
SOFA_LINK_CLASS(SimpleTesselatedTetraTopologicalMapping)
SOFA_LINK_CLASS(Tetra2TriangleTopologicalMapping)
SOFA_LINK_CLASS(Triangle2EdgeTopologicalMapping)
SOFA_LINK_CLASS(IdentityTopologicalMapping)


} // namespace component

} // namespace sofa
