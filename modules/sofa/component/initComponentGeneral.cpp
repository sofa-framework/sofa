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
#include <sofa/component/initComponentGeneral.h>


namespace sofa
{

namespace component
{


void initComponentGeneral()
{
    static bool first = true;
    if (first)
    {
        first = false;
    }
}

SOFA_LINK_CLASS(RayTraceDetection)
SOFA_LINK_CLASS(RayContact)
//SOFA_LINK_CLASS(ComponentMouseInteraction)
SOFA_LINK_CLASS(MouseInteractor)
//SOFA_LINK_CLASS(AttachBodyPerformer)
//SOFA_LINK_CLASS(FixParticlePerformer)
//SOFA_LINK_CLASS(InteractionPerformer)
//SOFA_LINK_CLASS(SuturePointPerformer)
SOFA_LINK_CLASS(LocalMinDistance)
SOFA_LINK_CLASS(LMDNewProximityIntersection)
SOFA_LINK_CLASS(BarycentricDistanceLMConstraintContact)
SOFA_LINK_CLASS(FrictionContact)
SOFA_LINK_CLASS(RotationFinder)
SOFA_LINK_CLASS(AspirationForceField)
SOFA_LINK_CLASS(BuoyantForceField)
SOFA_LINK_CLASS(ConicalForceField)
SOFA_LINK_CLASS(ConstantForceField)
SOFA_LINK_CLASS(EdgePressureForceField)
SOFA_LINK_CLASS(EllipsoidForceField)
SOFA_LINK_CLASS(LinearForceField)
SOFA_LINK_CLASS(OscillatingTorsionPressureForceField)
SOFA_LINK_CLASS(PlaneForceField)
SOFA_LINK_CLASS(SphereForceField)
SOFA_LINK_CLASS(SurfacePressureForceField)
SOFA_LINK_CLASS(TrianglePressureForceField)
SOFA_LINK_CLASS(VaccumSphereForceField)
//SOFA_LINK_CLASS(PenalityContactFrictionForceField)
SOFA_LINK_CLASS(Mesh2PointMechanicalMapping)
SOFA_LINK_CLASS(SimpleTesselatedTetraMechanicalMapping)
SOFA_LINK_CLASS(AttachConstraint)
SOFA_LINK_CLASS(FixedConstraint)
SOFA_LINK_CLASS(FixedPlaneConstraint)
SOFA_LINK_CLASS(FixedRotationConstraint)
SOFA_LINK_CLASS(FixedTranslationConstraint)
SOFA_LINK_CLASS(HermiteSplineConstraint)
SOFA_LINK_CLASS(LinearMovementConstraint)
SOFA_LINK_CLASS(LinearVelocityConstraint)
SOFA_LINK_CLASS(OscillatorConstraint)
SOFA_LINK_CLASS(ParabolicConstraint)
SOFA_LINK_CLASS(PartialFixedConstraint)
SOFA_LINK_CLASS(PartialLinearMovementConstraint)
SOFA_LINK_CLASS(AddFrameButtonSetting)
SOFA_LINK_CLASS(AttachBodyButtonSetting)
SOFA_LINK_CLASS(BackgroundSetting)
SOFA_LINK_CLASS(FixPickedParticleButtonSetting)
//SOFA_LINK_CLASS(MouseButtonSetting)
SOFA_LINK_CLASS(SofaDefaultPathSetting)
SOFA_LINK_CLASS(StatsSetting)
SOFA_LINK_CLASS(ViewerSetting)
SOFA_LINK_CLASS(Gravity)
//SOFA_LINK_CLASS(CoordinateSystem)
//SOFA_LINK_CLASS(GuidedCoordinateSystem)
SOFA_LINK_CLASS(BoxStiffSpringForceField)
SOFA_LINK_CLASS(InteractionEllipsoidForceField)
SOFA_LINK_CLASS(PenalityContactForceField)
SOFA_LINK_CLASS(RepulsiveSpringForceField)
SOFA_LINK_CLASS(UnilateralInteractionConstraint)
SOFA_LINK_CLASS(UncoupledConstraintCorrection)
SOFA_LINK_CLASS(PrecomputedConstraintCorrection)
SOFA_LINK_CLASS(LinearSolverConstraintCorrection)
SOFA_LINK_CLASS(LCPConstraintSolver)
//SOFA_LINK_CLASS(ConstraintSolverImpl)
SOFA_LINK_CLASS(DOFBlockerLMConstraint)
SOFA_LINK_CLASS(FixedLMConstraint)
SOFA_LINK_CLASS(DistanceLMContactConstraint)
SOFA_LINK_CLASS(DistanceLMConstraint)
SOFA_LINK_CLASS(LMConstraintSolver)
SOFA_LINK_CLASS(LMConstraintDirectSolver)
SOFA_LINK_CLASS(ArticulatedHierarchyController)
SOFA_LINK_CLASS(ArticulatedHierarchyBVHController)
//SOFA_LINK_CLASS(Controller)
SOFA_LINK_CLASS(EdgeSetController)
SOFA_LINK_CLASS(MechanicalStateController)
SOFA_LINK_CLASS(NullForceFeedback)
SOFA_LINK_CLASS(EnslavementForceFeedback)
SOFA_LINK_CLASS(LCPForceFeedback)
SOFA_LINK_CLASS(AverageCoord)
SOFA_LINK_CLASS(BoxROI)
SOFA_LINK_CLASS(PlaneROI)
SOFA_LINK_CLASS(SphereROI)
SOFA_LINK_CLASS(ExtrudeSurface)
SOFA_LINK_CLASS(ExtrudeQuadsAndGenerateHexas)
SOFA_LINK_CLASS(GenerateRigidMass)
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
//SOFA_LINK_CLASS(CompareState)
//SOFA_LINK_CLASS(CompareTopology)
SOFA_LINK_CLASS(DevAngleCollisionMonitor)
SOFA_LINK_CLASS(DevTensionMonitor)
SOFA_LINK_CLASS(DevMonitorManager)
SOFA_LINK_CLASS(ExtraMonitor)
SOFA_LINK_CLASS(Monitor)
//SOFA_LINK_CLASS(PauseAnimation)
SOFA_LINK_CLASS(PauseAnimationOnEvent)
SOFA_LINK_CLASS(WriteState)
SOFA_LINK_CLASS(WriteTopology)
SOFA_LINK_CLASS(EvalPointsDistance)
SOFA_LINK_CLASS(EvalSurfaceDistance)
SOFA_LINK_CLASS(VTKExporter)
SOFA_LINK_CLASS(OBJExporter)
SOFA_LINK_CLASS(MeshExporter)
SOFA_LINK_CLASS(OglModel)
SOFA_LINK_CLASS(OglViewport)
SOFA_LINK_CLASS(Light)
SOFA_LINK_CLASS(LightManager)
SOFA_LINK_CLASS(PointSplatModel)
SOFA_LINK_CLASS(OglRenderingSRGB)
SOFA_LINK_CLASS(ClipPlane)
SOFA_LINK_CLASS(InteractiveCamera)
//SOFA_LINK_CLASS(VisualStyle)
//SOFA_LINK_CLASS(OglAttribute)
SOFA_LINK_CLASS(OglShader)
//SOFA_LINK_CLASS(OglShaderMacro)
SOFA_LINK_CLASS(OglShaderVisualModel)
SOFA_LINK_CLASS(OglShadowShader)
SOFA_LINK_CLASS(OglTetrahedralModel)
SOFA_LINK_CLASS(OglTexture)
//SOFA_LINK_CLASS(OglVariable)
SOFA_LINK_CLASS(PostProcessManager)
SOFA_LINK_CLASS(SlicedVolumetricModel)
SOFA_LINK_CLASS(RecordedCamera)
SOFA_LINK_CLASS(FreeMotionAnimationLoop)
SOFA_LINK_CLASS(ShewchukPCGLinearSolver)
SOFA_LINK_CLASS(PCGLinearSolver)
SOFA_LINK_CLASS(JacobiPreconditioner)
SOFA_LINK_CLASS(BlockJacobiPreconditioner)
SOFA_LINK_CLASS(SSORPreconditioner)
SOFA_LINK_CLASS(LULinearSolver)
//SOFA_LINK_CLASS(NewMatVector)
//SOFA_LINK_CLASS(NewMatMatrix)
SOFA_LINK_CLASS(WarpPreconditioner)
SOFA_LINK_CLASS(PrecomputedWarpPreconditioner)
#ifdef SOFA_HAVE_CSPARSE
SOFA_LINK_CLASS(PrecomputedLinearSolver)
SOFA_LINK_CLASS(SparseCholeskySolver)
SOFA_LINK_CLASS(SparseLUSolver)
SOFA_LINK_CLASS(SparseLDLSolver)
SOFA_LINK_CLASS(SparseXXTSolver)
#endif
//SOFA_LINK_CLASS(DevMonitor)
//SOFA_LINK_CLASS(ContactDescription)
//SOFA_LINK_CLASS(ForceFeedback)
//SOFA_LINK_CLASS(NewMatCGLinearSolver)
//SOFA_LINK_CLASS(NewMatCholeskySolver)



} // namespace component

} // namespace sofa
