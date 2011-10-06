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
#include <sofa/component/initComponentAdvanced.h>


namespace sofa
{

namespace component
{


void initComponentAdvanced()
{
    static bool first = true;
    if (first)
    {
        first = false;
    }
}

SOFA_LINK_CLASS(DistanceGridCollisionModel)
SOFA_LINK_CLASS(EdgeRemoveContact)
SOFA_LINK_CLASS(CarvingManager)
SOFA_LINK_CLASS(ComplianceMatrixUpdateManager)
SOFA_LINK_CLASS(ComplianceMatrixUpdateManagerCarving)
SOFA_LINK_CLASS(CuttingManager)
//SOFA_LINK_CLASS(RayModel)
SOFA_LINK_CLASS(GraspingManager)
SOFA_LINK_CLASS(SuturingManager)
SOFA_LINK_CLASS(TetrahedronCuttingManager)
//SOFA_LINK_CLASS(TopologicalChangeManager)
SOFA_LINK_CLASS(TriangularFEMFractureManager)
//SOFA_LINK_CLASS(AddFramePerformer)
//SOFA_LINK_CLASS(InciseAlongPathPerformer)
//SOFA_LINK_CLASS(RemovePrimitivePerformer)
//SOFA_LINK_CLASS(BsplineFrictionContact)
SOFA_LINK_CLASS(BeamBsplineContactMapper)
//SOFA_LINK_CLASS(ImplicitSurfaceContainer)
SOFA_LINK_CLASS(InterpolatedImplicitSurface)
SOFA_LINK_CLASS(SpatialGridContainer)
SOFA_LINK_CLASS(DistanceGridForceField)
SOFA_LINK_CLASS(NonUniformHexahedralFEMForceFieldAndMass)
SOFA_LINK_CLASS(NonUniformHexahedronFEMForceFieldAndMass)
SOFA_LINK_CLASS(NonUniformHexahedronFEMForceFieldDensity)
SOFA_LINK_CLASS(ParticlesRepulsionForceField)
SOFA_LINK_CLASS(SPHFluidForceField)
SOFA_LINK_CLASS(AdhesiveSurfaceForceField)
SOFA_LINK_CLASS(MJEDTetrahedralForceField)
SOFA_LINK_CLASS(StandardTetrahedralFEMForceField)
SOFA_LINK_CLASS(TetrahedralBiquadraticSpringsForceField)
SOFA_LINK_CLASS(TetrahedralQuadraticSpringsForceField)
SOFA_LINK_CLASS(TetrahedralTotalLagrangianForceField)
SOFA_LINK_CLASS(NonUniformHexahedralFEMForceFieldAndMassCorrected)
SOFA_LINK_CLASS(ImplicitSurfaceMapping)
SOFA_LINK_CLASS(SPHFluidSurfaceMapping)
SOFA_LINK_CLASS(BeamBsplineMapping)
SOFA_LINK_CLASS(DiscreteBeamBsplineMapping)
SOFA_LINK_CLASS(ProjectionLineConstraint)
SOFA_LINK_CLASS(ProjectionPlaneConstraint)
SOFA_LINK_CLASS(DisplacementConstraint)
SOFA_LINK_CLASS(Fluid2D)
SOFA_LINK_CLASS(Fluid3D)
//SOFA_LINK_CLASS(Grid2D)
//SOFA_LINK_CLASS(Grid3D)
//SOFA_LINK_CLASS(Material)
SOFA_LINK_CLASS(PlasticMaterial)
//SOFA_LINK_CLASS(HyperelasticMaterial)
//SOFA_LINK_CLASS(BoyceAndArruda)
//SOFA_LINK_CLASS(STVenantKirchhoff)
//SOFA_LINK_CLASS(NeoHookean)
//SOFA_LINK_CLASS(MooneyRivlin)
//SOFA_LINK_CLASS(VerondaWestman)
//SOFA_LINK_CLASS(Costa)
//SOFA_LINK_CLASS(NeoHookeanIsotropicMJED)
//SOFA_LINK_CLASS(HyperelasticMaterialMJED)
//SOFA_LINK_CLASS(BoyceAndArrudaMJED)
//SOFA_LINK_CLASS(STVenantKirchhoffMJED)
//SOFA_LINK_CLASS(NeoHookeanMJED)
//SOFA_LINK_CLASS(MooneyRivlinMJED)
//SOFA_LINK_CLASS(VerondaWestmanMJED)
//SOFA_LINK_CLASS(CostaMJED)
//SOFA_LINK_CLASS(OgdenMJED)
SOFA_LINK_CLASS(GenericConstraintSolver)
SOFA_LINK_CLASS(StopperConstraint)
SOFA_LINK_CLASS(SlidingConstraint)
SOFA_LINK_CLASS(BilateralInteractionConstraint)
SOFA_LINK_CLASS(BeamConstraint)
SOFA_LINK_CLASS(ControllerVerification)
SOFA_LINK_CLASS(JointSpringController)
SOFA_LINK_CLASS(HandStateController)
SOFA_LINK_CLASS(LaparoscopicController)
SOFA_LINK_CLASS(RespirationController)
SOFA_LINK_CLASS(VMechanismsForceFeedback)
SOFA_LINK_CLASS(DynamicSparseGridGeometryAlgorithms)
SOFA_LINK_CLASS(DynamicSparseGridTopologyAlgorithms)
SOFA_LINK_CLASS(DynamicSparseGridTopologyContainer)
SOFA_LINK_CLASS(DynamicSparseGridTopologyModifier)
SOFA_LINK_CLASS(MultilevelHexahedronSetTopologyContainer)
SOFA_LINK_CLASS(SparseGridMultipleTopology)
SOFA_LINK_CLASS(SparseGridRamificationTopology)
SOFA_LINK_CLASS(MultilevelHexahedronSetGeometryAlgorithms)
SOFA_LINK_CLASS(MultilevelHexahedronSetTopologyAlgorithms)
SOFA_LINK_CLASS(MultilevelHexahedronSetTopologyModifier)
SOFA_LINK_CLASS(ParticleSink)
SOFA_LINK_CLASS(ParticleSource)
SOFA_LINK_CLASS(Hexa2TriangleTopologicalMapping)
SOFA_LINK_CLASS(MultilevelHexaTopologicalMapping)
SOFA_LINK_CLASS(MultilevelHexa2TriangleTopologicalMapping)
SOFA_LINK_CLASS(HexahedronCompositeFEMForceFieldAndMass)
SOFA_LINK_CLASS(HexahedronCompositeFEMMapping)
SOFA_LINK_CLASS(ConstraintAnimationLoop)
//SOFA_LINK_CLASS(EigenMatrixManipulator)
SOFA_LINK_CLASS(SVDLinearSolver)
SOFA_LINK_CLASS(SparseTAUCSSolver)
SOFA_LINK_CLASS(IncompleteTAUCSSolver)
SOFA_LINK_CLASS(SparseTAUCSLLtSolver)
//SOFA_LINK_CLASS(SparsePARDISOSolver)
//SOFA_LINK_CLASS(Ogden)
//SOFA_LINK_CLASS(BaseMaterial)
SOFA_LINK_CLASS(FractureManager)



} // namespace component

} // namespace sofa
