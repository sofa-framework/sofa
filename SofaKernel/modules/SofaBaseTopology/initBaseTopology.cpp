/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaBaseTopology/initBaseTopology.h>


namespace sofa
{

namespace component
{


void initBaseTopology()
{
    static bool first = true;
    if (first)
    {
        first = false;
    }
}

SOFA_LINK_CLASS(EdgeSetGeometryAlgorithms)
SOFA_LINK_CLASS(EdgeSetTopologyAlgorithms)
SOFA_LINK_CLASS(EdgeSetTopologyContainer)
SOFA_LINK_CLASS(EdgeSetTopologyModifier)
SOFA_LINK_CLASS(GridTopology)
SOFA_LINK_CLASS(HexahedronSetGeometryAlgorithms)
SOFA_LINK_CLASS(HexahedronSetTopologyAlgorithms)
SOFA_LINK_CLASS(HexahedronSetTopologyContainer)
SOFA_LINK_CLASS(HexahedronSetTopologyModifier)
SOFA_LINK_CLASS(MeshTopology)
SOFA_LINK_CLASS(PointSetGeometryAlgorithms)
SOFA_LINK_CLASS(PointSetTopologyAlgorithms)
SOFA_LINK_CLASS(PointSetTopologyContainer)
SOFA_LINK_CLASS(PointSetTopologyModifier)
SOFA_LINK_CLASS(QuadSetGeometryAlgorithms)
SOFA_LINK_CLASS(QuadSetTopologyAlgorithms)
SOFA_LINK_CLASS(QuadSetTopologyContainer)
SOFA_LINK_CLASS(QuadSetTopologyModifier)
SOFA_LINK_CLASS(RegularGridTopology)
SOFA_LINK_CLASS(SparseGridTopology)
SOFA_LINK_CLASS(TetrahedronSetGeometryAlgorithms)
SOFA_LINK_CLASS(TetrahedronSetTopologyAlgorithms)
SOFA_LINK_CLASS(TetrahedronSetTopologyContainer)
SOFA_LINK_CLASS(TetrahedronSetTopologyModifier)
SOFA_LINK_CLASS(TriangleSetGeometryAlgorithms)
SOFA_LINK_CLASS(TriangleSetTopologyAlgorithms)
SOFA_LINK_CLASS(TriangleSetTopologyContainer)
SOFA_LINK_CLASS(TriangleSetTopologyModifier)

} // namespace component

} // namespace sofa
