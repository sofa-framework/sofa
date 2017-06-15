/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#include <SofaGeneralEngine/initGeneralEngine.h>


namespace sofa
{

namespace component
{


void initGeneralEngine()
{
    static bool first = true;
    if (first)
    {
        first = false;
    }
}

SOFA_LINK_CLASS(AverageCoord)
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
SOFA_LINK_CLASS(MergeROIs)
SOFA_LINK_CLASS(MeshBarycentricMapperEngine)
SOFA_LINK_CLASS(TransformPosition)
SOFA_LINK_CLASS(TransformEngine)
SOFA_LINK_CLASS(PointsFromIndices)
SOFA_LINK_CLASS(ValuesFromIndices)
SOFA_LINK_CLASS(IndicesFromValues)
SOFA_LINK_CLASS(IndexValueMapper)
SOFA_LINK_CLASS(ROIValueMapper)
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
SOFA_LINK_CLASS(MeshClosingEngine)
SOFA_LINK_CLASS(MeshSubsetEngine)
SOFA_LINK_CLASS(MeshSampler)
SOFA_LINK_CLASS(MeshSplittingEngine)

} // namespace component

} // namespace sofa
