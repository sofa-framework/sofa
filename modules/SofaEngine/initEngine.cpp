/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/helper/system/config.h>
#include <SofaEngine/initEngine.h>


namespace sofa
{

namespace component
{


void initEngine()
{
    static bool first = true;
    if (first)
    {
        first = false;
    }
}

SOFA_LINK_CLASS(AverageCoord)
SOFA_LINK_CLASS(BoxROI)
SOFA_LINK_CLASS(PlaneROI)
SOFA_LINK_CLASS(SphereROI)
SOFA_LINK_CLASS(DilateEngine)
SOFA_LINK_CLASS(ExtrudeSurface)
SOFA_LINK_CLASS(ExtrudeQuadsAndGenerateHexas)
SOFA_LINK_CLASS(GenerateRigidMass)
SOFA_LINK_CLASS(GenerateCylinder)
SOFA_LINK_CLASS(GroupFilterYoungModulus)
SOFA_LINK_CLASS(MergeMeshes)
SOFA_LINK_CLASS(MergePoints)
SOFA_LINK_CLASS(MergeSets)
SOFA_LINK_CLASS(MeshBarycentricMapperEngine)
SOFA_LINK_CLASS(TransformPosition)
SOFA_LINK_CLASS(TransformEngine)
SOFA_LINK_CLASS(PointsFromIndices)
SOFA_LINK_CLASS(ValuesFromIndices)
SOFA_LINK_CLASS(IndicesFromValues)
SOFA_LINK_CLASS(IndexValueMapper)
SOFA_LINK_CLASS(JoinPoints)
SOFA_LINK_CLASS(MapIndices)
SOFA_LINK_CLASS(RandomPointDistributionInSurface)
SOFA_LINK_CLASS(Spiral)
SOFA_LINK_CLASS(Vertex2Frame)
SOFA_LINK_CLASS(TextureInterpolation)
SOFA_LINK_CLASS(SubsetTopology)
SOFA_LINK_CLASS(RigidToQuatEngine)
SOFA_LINK_CLASS(QuatToRigidEngine)
SOFA_LINK_CLASS(ValuesFromPositions)
SOFA_LINK_CLASS(NormalsFromPoints)
SOFA_LINK_CLASS(ClusteringEngine)
SOFA_LINK_CLASS(ShapeMatching)
SOFA_LINK_CLASS(ProximityROI)

} // namespace component

} // namespace sofa
