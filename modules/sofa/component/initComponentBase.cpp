/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/component/initComponentBase.h>


namespace sofa
{

namespace component
{


void initComponentBase()
{
    static bool first = true;
    if (first)
    {
        first = false;
    }
}

//SOFA_LINK_CLASS(RayTriangleIntersection)
SOFA_LINK_CLASS(DefaultPipeline)
SOFA_LINK_CLASS(Sphere)
//SOFA_LINK_CLASS(SphereModel)
SOFA_LINK_CLASS(Cube)
//SOFA_LINK_CLASS(CubeModel)
SOFA_LINK_CLASS(DiscreteIntersection)
//SOFA_LINK_CLASS(BruteForceDetection)
//SOFA_LINK_CLASS(BaseContactMapper)
SOFA_LINK_CLASS(DefaultContactManager)
SOFA_LINK_CLASS(Point)
//SOFA_LINK_CLASS(PointModel)
SOFA_LINK_CLASS(Line)
//SOFA_LINK_CLASS(LineModel)
SOFA_LINK_CLASS(Triangle)
//SOFA_LINK_CLASS(TriangleModel)
SOFA_LINK_CLASS(TetrahedronModel)
SOFA_LINK_CLASS(SharpLineModel)
SOFA_LINK_CLASS(SpatialGridPointModel)
SOFA_LINK_CLASS(SphereTreeModel)
//SOFA_LINK_CLASS(TriangleOctree)
//SOFA_LINK_CLASS(TriangleOctreeModel)
//SOFA_LINK_CLASS(RayModel)
//SOFA_LINK_CLASS(BsplineModel)
SOFA_LINK_CLASS(LineLocalMinDistanceFilter)
//SOFA_LINK_CLASS(LocalMinDistanceFilter)
SOFA_LINK_CLASS(PointLocalMinDistanceFilter)
//SOFA_LINK_CLASS(TriangleLocalMinDistanceFilter)
SOFA_LINK_CLASS(MappedObject)
SOFA_LINK_CLASS(MechanicalObject)
//SOFA_LINK_CLASS(DistanceGrid)
//SOFA_LINK_CLASS(MechanicalObjectTasks)
//SOFA_LINK_CLASS(AddMToMatrixFunctor)
SOFA_LINK_CLASS(DiagonalMass)
SOFA_LINK_CLASS(UniformMass)
SOFA_LINK_CLASS(BarycentricMapping)
SOFA_LINK_CLASS(IdentityMapping)
SOFA_LINK_CLASS(SubsetMapping)
//SOFA_LINK_CLASS(CommonAlgorithms)
SOFA_LINK_CLASS(CubeTopology)
SOFA_LINK_CLASS(CylinderGridTopology)
//SOFA_LINK_CLASS(EdgeData)
SOFA_LINK_CLASS(EdgeSetGeometryAlgorithms)
SOFA_LINK_CLASS(EdgeSetTopologyAlgorithms)
//SOFA_LINK_CLASS(EdgeSetTopologyChange)
SOFA_LINK_CLASS(EdgeSetTopologyContainer)
SOFA_LINK_CLASS(EdgeSetTopologyModifier)
//SOFA_LINK_CLASS(EdgeSetTopologyEngine)
//SOFA_LINK_CLASS(EdgeSubsetData)
SOFA_LINK_CLASS(GridTopology)
//SOFA_LINK_CLASS(HexahedronData)
SOFA_LINK_CLASS(HexahedronSetGeometryAlgorithms)
SOFA_LINK_CLASS(HexahedronSetTopologyAlgorithms)
//SOFA_LINK_CLASS(HexahedronSetTopologyChange)
SOFA_LINK_CLASS(HexahedronSetTopologyContainer)
SOFA_LINK_CLASS(HexahedronSetTopologyModifier)
//SOFA_LINK_CLASS(HexahedronSetTopologyEngine)
SOFA_LINK_CLASS(MeshTopology)
//SOFA_LINK_CLASS(PointData)
//SOFA_LINK_CLASS(PointSetTopologyEngine)
SOFA_LINK_CLASS(PointSetGeometryAlgorithms)
SOFA_LINK_CLASS(PointSetTopologyAlgorithms)
//SOFA_LINK_CLASS(PointSetTopologyChange)
SOFA_LINK_CLASS(PointSetTopologyContainer)
SOFA_LINK_CLASS(PointSetTopologyModifier)
//SOFA_LINK_CLASS(PointSubset)
//SOFA_LINK_CLASS(QuadData)
SOFA_LINK_CLASS(QuadSetGeometryAlgorithms)
SOFA_LINK_CLASS(QuadSetTopologyAlgorithms)
//SOFA_LINK_CLASS(QuadSetTopologyChange)
SOFA_LINK_CLASS(QuadSetTopologyContainer)
SOFA_LINK_CLASS(QuadSetTopologyModifier)
//SOFA_LINK_CLASS(QuadSetTopologyEngine)
SOFA_LINK_CLASS(RegularGridTopology)
SOFA_LINK_CLASS(SparseGridTopology)
//SOFA_LINK_CLASS(TetrahedronData)
SOFA_LINK_CLASS(TetrahedronSetGeometryAlgorithms)
SOFA_LINK_CLASS(TetrahedronSetTopologyAlgorithms)
//SOFA_LINK_CLASS(TetrahedronSetTopologyChange)
SOFA_LINK_CLASS(TetrahedronSetTopologyContainer)
SOFA_LINK_CLASS(TetrahedronSetTopologyModifier)
//SOFA_LINK_CLASS(TetrahedronSetTopologyEngine)
//SOFA_LINK_CLASS(TopologyChangedEvent)
//SOFA_LINK_CLASS(TriangleData)
SOFA_LINK_CLASS(TriangleSetGeometryAlgorithms)
SOFA_LINK_CLASS(TriangleSetTopologyAlgorithms)
//SOFA_LINK_CLASS(TriangleSetTopologyChange)
SOFA_LINK_CLASS(TriangleSetTopologyContainer)
SOFA_LINK_CLASS(TriangleSetTopologyModifier)
//SOFA_LINK_CLASS(TriangleSetTopologyEngine)
//SOFA_LINK_CLASS(TriangleSubsetData)
//SOFA_LINK_CLASS(BaseCamera)
SOFA_LINK_CLASS(VisualModelImpl)
//SOFA_LINK_CLASS(DefaultMasterSolver)
SOFA_LINK_CLASS(MultiStepAnimationLoop)
SOFA_LINK_CLASS(MultiTagAnimationLoop)
SOFA_LINK_CLASS(CGLinearSolver)
SOFA_LINK_CLASS(CholeskySolver)
SOFA_LINK_CLASS(BTDLinearSolver)
//SOFA_LINK_CLASS(FullVector)
//SOFA_LINK_CLASS(FullMatrix)
//SOFA_LINK_CLASS(DiagonalMatrix)
//SOFA_LINK_CLASS(SparseMatrix)
//SOFA_LINK_CLASS(CompressedRowSparseMatrix)
//SOFA_LINK_CLASS(GraphScatteredTypes)
//SOFA_LINK_CLASS(DefaultMultiMatrixAccessor)
//SOFA_LINK_CLASS(MatrixLinearSolver)
//SOFA_LINK_CLASS(ParallelMatrixLinearSolver)
//SOFA_LINK_CLASS(MatrixExpr)
SOFA_LINK_CLASS(GenerateBenchSolver)
//SOFA_LINK_CLASS(matrix_bloc_traits)
SOFA_LINK_CLASS(Ray)


} // namespace component

} // namespace sofa
