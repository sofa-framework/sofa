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
#include <sofa/component/initComponentCommon.h>


namespace sofa
{

namespace component
{


void initComponentCommon()
{
    static bool first = true;
    if (first)
    {
        first = false;
    }
}

//SOFA_LINK_CLASS(DefaultCollisionGroupManager)
SOFA_LINK_CLASS(MinProximityIntersection)
SOFA_LINK_CLASS(NewProximityIntersection)
SOFA_LINK_CLASS(BarycentricPenalityContact)
SOFA_LINK_CLASS(BarycentricContactMapper)
SOFA_LINK_CLASS(IdentityContactMapper)
SOFA_LINK_CLASS(SubsetContactMapper)
//SOFA_LINK_CLASS(SolverMerger)
SOFA_LINK_CLASS(ArticulatedHierarchyContainer)
SOFA_LINK_CLASS(BeamFEMForceField)
SOFA_LINK_CLASS(HexahedralFEMForceField)
SOFA_LINK_CLASS(HexahedralFEMForceFieldAndMass)
SOFA_LINK_CLASS(HexahedronFEMForceField)
SOFA_LINK_CLASS(HexahedronFEMForceFieldAndMass)
SOFA_LINK_CLASS(QuadularBendingSprings)
SOFA_LINK_CLASS(RestShapeSpringsForceField)
SOFA_LINK_CLASS(TetrahedralCorotationalFEMForceField)
SOFA_LINK_CLASS(TetrahedronFEMForceField)
SOFA_LINK_CLASS(TriangularAnisotropicFEMForceField)
SOFA_LINK_CLASS(TriangularBendingSprings)
SOFA_LINK_CLASS(TriangularBiquadraticSpringsForceField)
SOFA_LINK_CLASS(TriangleFEMForceField)
SOFA_LINK_CLASS(TriangularFEMForceField)
SOFA_LINK_CLASS(TriangularQuadraticSpringsForceField)
SOFA_LINK_CLASS(TriangularTensorMassForceField)
SOFA_LINK_CLASS(ArticulatedSystemMapping)
SOFA_LINK_CLASS(LaparoscopicRigidMapping)
SOFA_LINK_CLASS(LineSetSkinningMapping)
SOFA_LINK_CLASS(RigidMapping)
SOFA_LINK_CLASS(RigidRigidMapping)
SOFA_LINK_CLASS(SkinningMapping)
SOFA_LINK_CLASS(MeshGmshLoader)
SOFA_LINK_CLASS(MeshObjLoader)
SOFA_LINK_CLASS(MeshOffLoader)
SOFA_LINK_CLASS(MeshTrianLoader)
SOFA_LINK_CLASS(MeshVTKLoader)
SOFA_LINK_CLASS(MeshSTLLoader)
SOFA_LINK_CLASS(MeshXspLoader)
SOFA_LINK_CLASS(OffSequenceLoader)
SOFA_LINK_CLASS(FrameSpringForceField)
SOFA_LINK_CLASS(JointSpringForceField)
SOFA_LINK_CLASS(MeshSpringForceField)
SOFA_LINK_CLASS(QuadBendingSprings)
SOFA_LINK_CLASS(RegularGridSpringForceField)
SOFA_LINK_CLASS(SpringForceField)
SOFA_LINK_CLASS(StiffSpringForceField)
SOFA_LINK_CLASS(TriangleBendingSprings)
SOFA_LINK_CLASS(VectorSpringForceField)
SOFA_LINK_CLASS(InputEventReader)
SOFA_LINK_CLASS(ReadState)
SOFA_LINK_CLASS(ReadTopology)
SOFA_LINK_CLASS(CentralDifferenceSolver)
//SOFA_LINK_CLASS(EulerSolver)
SOFA_LINK_CLASS(EulerImplicitSolver)
//SOFA_LINK_CLASS(RungeKutta2Solver)
//SOFA_LINK_CLASS(RungeKutta4Solver)
SOFA_LINK_CLASS(StaticSolver)
//SOFA_LINK_CLASS(LengthContainer)
//SOFA_LINK_CLASS(PoissonContainer)
//SOFA_LINK_CLASS(RadiusContainer)
//SOFA_LINK_CLASS(StiffnessContainer)



} // namespace component

} // namespace sofa
