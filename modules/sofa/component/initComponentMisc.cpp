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
#include <sofa/component/initComponentMisc.h>


namespace sofa
{

namespace component
{


void initComponentMisc()
{
    static bool first = true;
    if (first)
    {
        first = false;
    }
}

//SOFA_LINK_CLASS(ParallelCollisionPipeline)
//SOFA_LINK_CLASS(TriangleModelInRegularGrid)
//SOFA_LINK_CLASS(ContinuousTriangleIntersection)
//SOFA_LINK_CLASS(RigidContactMapper)
//SOFA_LINK_CLASS(RuleBasedContactMapper)
SOFA_LINK_CLASS(DiscreteBeamBsplineMapping)
SOFA_LINK_CLASS(TreeCollisionGroupManager)
SOFA_LINK_CLASS(MatrixMass)
SOFA_LINK_CLASS(MeshMatrixMass)
SOFA_LINK_CLASS(FastTetrahedralCorotationalForceField)
SOFA_LINK_CLASS(LennardJonesForceField)
SOFA_LINK_CLASS(TetrahedralTensorMassForceField)
SOFA_LINK_CLASS(WashingMachineForceField)
SOFA_LINK_CLASS(BeamLinearMapping)
SOFA_LINK_CLASS(CenterPointMechanicalMapping)
SOFA_LINK_CLASS(CenterOfMassMapping)
SOFA_LINK_CLASS(CenterOfMassMultiMapping)
SOFA_LINK_CLASS(CenterOfMassMulti2Mapping)
SOFA_LINK_CLASS(CurveMapping)
SOFA_LINK_CLASS(ExternalInterpolationMapping)
SOFA_LINK_CLASS(SubsetMultiMapping)
SOFA_LINK_CLASS(TubularMapping)
SOFA_LINK_CLASS(VoidMapping)
SOFA_LINK_CLASS(Distances)
//SOFA_LINK_CLASS(IndexedMapTopology)
SOFA_LINK_CLASS(MeshTetraStuffing)
SOFA_LINK_CLASS(TopologicalChangeProcessor)
//SOFA_LINK_CLASS(Triplet)
//SOFA_LINK_CLASS(SubdivisionCell)
//SOFA_LINK_CLASS(TopologyChangeTest)
//SOFA_LINK_CLASS(BoundingBox)
//SOFA_LINK_CLASS(TagRule)
//SOFA_LINK_CLASS(ParallelCGLinearSolver)
//SOFA_LINK_CLASS(DampVelocitySolver)
SOFA_LINK_CLASS(NewmarkImplicitSolver)

#ifdef SOFA_DEV
SOFA_LINK_CLASS(ContinuousIntersection)
SOFA_LINK_CLASS(TestDetection)
SOFA_LINK_CLASS(BarycentricStickContact)
SOFA_LINK_CLASS(BglCollisionGroupManager)
SOFA_LINK_CLASS(ArborisDescription)
SOFA_LINK_CLASS(ShapeMatchingForceField)
//SOFA_LINK_CLASS(VectorField)
SOFA_LINK_CLASS(ArborisMapping)
SOFA_LINK_CLASS(CircumcenterMapping)
SOFA_LINK_CLASS(DeformableOnRigidFrameMapping)
SOFA_LINK_CLASS(PCAOnRigidFrameMapping)
SOFA_LINK_CLASS(Triangle2DFEMForceField)
SOFA_LINK_CLASS(TriangleBendingFEMForceField)
//SOFA_LINK_CLASS(FluidSolidInteractionForceField)
SOFA_LINK_CLASS(DistanceOnGrid)
SOFA_LINK_CLASS(Edge2TetraTopologicalMapping)
SOFA_LINK_CLASS(TriangleSubdivisionTopologicalMapping)
SOFA_LINK_CLASS(SpringIt)
SOFA_LINK_CLASS(GraphScenePartionner)
SOFA_LINK_CLASS(MappedBeamToTetraForceField)
//SOFA_LINK_CLASS(FlowVisualModel)
SOFA_LINK_CLASS(NewtonEulerImplicit)
#endif

} // namespace component

} // namespace sofa
