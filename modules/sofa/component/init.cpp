/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/core/ObjectFactory.h>
#include <iostream>

namespace sofa
{

namespace component
{


void init()
{
    static bool first = true;
    if (first)
    {
//         std::cout << "Sofa components initialized."<<std::endl;

        //std::ofstream ofile("sofa-classes.html");
        //ofile << "<html><body>\n";
        //sofa::core::ObjectFactory::getInstance()->dumpHTML(ofile);
        //ofile << "</body></html>\n";
        first = false;
    }
}

} // namespace component

} // namespace sofa

////////// BEGIN CLASS LIST //////////
SOFA_LINK_CLASS(ArticulatedHierarchyContainer)
SOFA_LINK_CLASS(ArticulatedHierarchyController)
SOFA_LINK_CLASS(ArticulatedHierarchyBVHController)
SOFA_LINK_CLASS(ArticulatedSystemMapping)
SOFA_LINK_CLASS(Articulation)
SOFA_LINK_CLASS(ArticulationCenter)
SOFA_LINK_CLASS(AttachConstraint)
SOFA_LINK_CLASS(Attribute)
SOFA_LINK_CLASS(BarycentricMapping)
SOFA_LINK_CLASS(BarycentricContactMapper)
SOFA_LINK_CLASS(BarycentricPenalityContact)
SOFA_LINK_CLASS(BeamFEMForceField)
SOFA_LINK_CLASS(BeamLinearMapping)
SOFA_LINK_CLASS(BruteForce)
SOFA_LINK_CLASS(CarvingManager)
SOFA_LINK_CLASS(CenterPointMechanicalMapping)
SOFA_LINK_CLASS(CenterPointTopologicalMapping)
SOFA_LINK_CLASS(CenterOfMassMapping)
SOFA_LINK_CLASS(ClipPlane)
SOFA_LINK_CLASS(CurveMapping)
SOFA_LINK_CLASS(CentralDifferenceSolver)
SOFA_LINK_CLASS(CGImplicit)
SOFA_LINK_CLASS(CGLinearSolver)
SOFA_LINK_CLASS(CylinderGridTopology)
SOFA_LINK_CLASS(DefaultCollisionGroupManager)
SOFA_LINK_CLASS(DefaultContactManager)
SOFA_LINK_CLASS(RuleBasedContactManager)
SOFA_LINK_CLASS(DefaultMasterSolver)
SOFA_LINK_CLASS(ContinuousIntersection)
SOFA_LINK_CLASS(ConstantForceField)
SOFA_LINK_CLASS(BoxConstantForceField)
SOFA_LINK_CLASS(UncoupledConstraintCorrection)
SOFA_LINK_CLASS(CoordinateSystem)
SOFA_LINK_CLASS(Cube)
SOFA_LINK_CLASS(CubeTopology)
SOFA_LINK_CLASS(DiagonalMass)
SOFA_LINK_CLASS(DiscreteIntersection)
SOFA_LINK_CLASS(DistanceGridCollisionModel)
SOFA_LINK_CLASS(EdgePressureForceField)
SOFA_LINK_CLASS(EdgeSetController)
SOFA_LINK_CLASS(ManifoldEdgeSetGeometryAlgorithms)
SOFA_LINK_CLASS(ManifoldEdgeSetTopologyAlgorithms)
SOFA_LINK_CLASS(ManifoldEdgeSetTopologyContainer)
SOFA_LINK_CLASS(ManifoldEdgeSetTopologyModifier)
SOFA_LINK_CLASS(EdgeSetGeometryAlgorithms)
SOFA_LINK_CLASS(EdgeSetTopologyAlgorithms)
SOFA_LINK_CLASS(EdgeSetTopologyContainer)
SOFA_LINK_CLASS(EdgeSetTopologyModifier)
SOFA_LINK_CLASS(EllipsoidForceField)
SOFA_LINK_CLASS(Euler)
SOFA_LINK_CLASS(EulerImplicitSolver)
SOFA_LINK_CLASS(FixedConstraint)
SOFA_LINK_CLASS(FixedPlaneConstraint)
SOFA_LINK_CLASS(Fluid2D)
SOFA_LINK_CLASS(Fluid3D)
SOFA_LINK_CLASS(FrameSpringForceField)
SOFA_LINK_CLASS(FrictionContact)
SOFA_LINK_CLASS(GNode)
SOFA_LINK_CLASS(Gravity)
SOFA_LINK_CLASS(GridTopology)
SOFA_LINK_CLASS(HermiteSplineConstraint)
SOFA_LINK_CLASS(IdentityMapping)
SOFA_LINK_CLASS(ImplicitSurfaceMapping)
SOFA_LINK_CLASS(InputEventReader)
SOFA_LINK_CLASS(InteractionEllipsoidForceField)
SOFA_LINK_CLASS(JointSpringForceField)
SOFA_LINK_CLASS(LaparoscopicRigidMapping)
SOFA_LINK_CLASS(LennardJonesForceField)
SOFA_LINK_CLASS(Line)
SOFA_LINK_CLASS(LineSetSkinningMapping)
SOFA_LINK_CLASS(LinearSolverConstraintCorrection)
SOFA_LINK_CLASS(LocalMinDistance)
SOFA_LINK_CLASS(LULinearSolver)
SOFA_LINK_CLASS(BTDLinearSolver)
SOFA_LINK_CLASS(MappedObject)
SOFA_LINK_CLASS(MasterContactSolver)
SOFA_LINK_CLASS(MechanicalObject)
SOFA_LINK_CLASS(MechanicalStateController)
SOFA_LINK_CLASS(Mesh2PointMechanicalMapping)
SOFA_LINK_CLASS(Mesh2PointTopologicalMapping)
SOFA_LINK_CLASS(MeshLoader)
SOFA_LINK_CLASS(MeshSpringForceField)
SOFA_LINK_CLASS(MeshTopology)
SOFA_LINK_CLASS(MinProximityIntersection)
SOFA_LINK_CLASS(Monitor)
SOFA_LINK_CLASS(MultiStepMasterSolver)
SOFA_LINK_CLASS(NewProximityIntersection)
SOFA_LINK_CLASS(NewmarkImplicitSolver)
SOFA_LINK_CLASS(Node)
SOFA_LINK_CLASS(Object)
SOFA_LINK_CLASS(OglModel)
SOFA_LINK_CLASS(OscillatorConstraint)
SOFA_LINK_CLASS(ParabolicConstraint)
SOFA_LINK_CLASS(ParticleSink)
SOFA_LINK_CLASS(ParticleSource)
SOFA_LINK_CLASS(PenalityContactForceField)
SOFA_LINK_CLASS(DefaultPipeline)
SOFA_LINK_CLASS(PlaneForceField)
SOFA_LINK_CLASS(Point)
SOFA_LINK_CLASS(PointSetGeometryAlgorithms)
SOFA_LINK_CLASS(PointSetTopologyAlgorithms)
SOFA_LINK_CLASS(PointSetTopologyContainer)
SOFA_LINK_CLASS(PointSetTopologyModifier)
SOFA_LINK_CLASS(PrecomputedConstraintCorrection)
SOFA_LINK_CLASS(SimpleTesselatedTetraMechanicalMapping)
SOFA_LINK_CLASS(SimpleTesselatedTetraTopologicalMapping)
SOFA_LINK_CLASS(TriangleSetGeometryAlgorithms)
SOFA_LINK_CLASS(TriangleSetTopologyAlgorithms)
SOFA_LINK_CLASS(TriangleSetTopologyContainer)
SOFA_LINK_CLASS(TriangleSetTopologyModifier)
SOFA_LINK_CLASS(QuadSetGeometryAlgorithms)
SOFA_LINK_CLASS(QuadSetTopologyAlgorithms)
SOFA_LINK_CLASS(QuadSetTopologyContainer)
SOFA_LINK_CLASS(QuadSetTopologyModifier)
SOFA_LINK_CLASS(HexahedronSetGeometryAlgorithms)
SOFA_LINK_CLASS(HexahedronSetTopologyAlgorithms)
SOFA_LINK_CLASS(HexahedronSetTopologyContainer)
SOFA_LINK_CLASS(HexahedronSetTopologyModifier)
SOFA_LINK_CLASS(TetrahedronSetGeometryAlgorithms)
SOFA_LINK_CLASS(TetrahedronSetTopologyAlgorithms)
SOFA_LINK_CLASS(TetrahedronSetTopologyContainer)
SOFA_LINK_CLASS(TetrahedronSetTopologyModifier)
SOFA_LINK_CLASS(Ray)
SOFA_LINK_CLASS(RayContact)
SOFA_LINK_CLASS(RayTraceDetection)
SOFA_LINK_CLASS(RegularGridSpringForceField)
SOFA_LINK_CLASS(RegularGridTopology)
SOFA_LINK_CLASS(RepulsiveSpringForceField)
SOFA_LINK_CLASS(RigidMapping)
SOFA_LINK_CLASS(RigidRigidMapping)
SOFA_LINK_CLASS(RungeKutta4)
SOFA_LINK_CLASS(BoxConstraint)
SOFA_LINK_CLASS(BoxStiffSpringForceField)
SOFA_LINK_CLASS(SkinningMapping)
SOFA_LINK_CLASS(SparseGridTopology)
SOFA_LINK_CLASS(SpatialGridContainer)
SOFA_LINK_CLASS(SpatialGridPointModel)
SOFA_LINK_CLASS(HexahedronFEMForceField)
SOFA_LINK_CLASS(HexahedralFEMForceField)
SOFA_LINK_CLASS(Sphere)
SOFA_LINK_CLASS(SphereForceField)
SOFA_LINK_CLASS(ConicalForceField)
SOFA_LINK_CLASS(SPHFluidForceField)
SOFA_LINK_CLASS(SPHFluidSurfaceMapping)
SOFA_LINK_CLASS(SpringForceField)
SOFA_LINK_CLASS(StaticSolver)
SOFA_LINK_CLASS(StiffSpringForceField)
SOFA_LINK_CLASS(SubsetMapping)
SOFA_LINK_CLASS(TubularMapping)
SOFA_LINK_CLASS(Edge2QuadTopologicalMapping)
SOFA_LINK_CLASS(Triangle2EdgeTopologicalMapping)
SOFA_LINK_CLASS(Quad2TriangleTopologicalMapping)
SOFA_LINK_CLASS(Tetra2TriangleTopologicalMapping)
SOFA_LINK_CLASS(Hexa2QuadTopologicalMapping)
SOFA_LINK_CLASS(TetrahedronFEMForceField)
SOFA_LINK_CLASS(TetrahedronModel)
SOFA_LINK_CLASS(TetrahedralTensorMassForceField)
SOFA_LINK_CLASS(TetrahedralCorotationalFEMForceField)
SOFA_LINK_CLASS(Triangle)
SOFA_LINK_CLASS(TriangleBendingSprings)
SOFA_LINK_CLASS(TriangularBendingSprings)
SOFA_LINK_CLASS(TrianglePressureForceField)
SOFA_LINK_CLASS(TriangleFEMForceField)
SOFA_LINK_CLASS(TriangularFEMForceField)
SOFA_LINK_CLASS(TriangularAnisotropicFEMForceField)
SOFA_LINK_CLASS(TriangularQuadraticSpringsForceField)
SOFA_LINK_CLASS(TriangularBiquadraticSpringsForceField)
SOFA_LINK_CLASS(TriangularTensorMassForceField)
SOFA_LINK_CLASS(LinearMovementConstraint)
SOFA_LINK_CLASS(UniformMass)
SOFA_LINK_CLASS(Data)
SOFA_LINK_CLASS(SphereTreeModel)
SOFA_LINK_CLASS(ReadState)
SOFA_LINK_CLASS(WriteState)
SOFA_LINK_CLASS(QuadBendingSprings)
SOFA_LINK_CLASS(QuadularBendingSprings)
SOFA_LINK_CLASS(DirectionalLight)
SOFA_LINK_CLASS(PositionalLight)
SOFA_LINK_CLASS(SpotLight)
SOFA_LINK_CLASS(LightManager)
SOFA_LINK_CLASS(VoxelGridLoader)
SOFA_LINK_CLASS(SurfacePressureForceField)

#ifdef SOFA_HAVE_GLEW
SOFA_LINK_CLASS(OglShader)
SOFA_LINK_CLASS(OglTexture2D)
SOFA_LINK_CLASS(OglIntVariable)
SOFA_LINK_CLASS(OglInt2Variable)
SOFA_LINK_CLASS(OglInt3Variable)
SOFA_LINK_CLASS(OglInt4Variable)
SOFA_LINK_CLASS(OglFloatVariable)
SOFA_LINK_CLASS(OglFloat2Variable)
SOFA_LINK_CLASS(OglFloat3Variable)
SOFA_LINK_CLASS(OglFloat4Variable)
SOFA_LINK_CLASS(OglIntVectorVariable)
SOFA_LINK_CLASS(OglIntVector2Variable)
SOFA_LINK_CLASS(OglIntVector3Variable)
SOFA_LINK_CLASS(OglIntVector4Variable)
SOFA_LINK_CLASS(OglFloatVectorVariable)
SOFA_LINK_CLASS(OglFloatVector2Variable)
SOFA_LINK_CLASS(OglFloatVector3Variable)
SOFA_LINK_CLASS(OglFloatVector4Variable)
SOFA_LINK_CLASS(OglShaderDefineMacro)
SOFA_LINK_CLASS(OglTetrahedralModel)
#endif

SOFA_LINK_CLASS(VoidMapping)

#ifdef SOFA_DEV

// collision
//SOFA_LINK_CLASS(BarycentricLagrangianMultiplierContact)
SOFA_LINK_CLASS(BarycentricStickContact)
SOFA_LINK_CLASS(CuttingManager)
SOFA_LINK_CLASS(EdgeRemoveContact)
SOFA_LINK_CLASS(FractureManager)
SOFA_LINK_CLASS(GraspingManager)
SOFA_LINK_CLASS(SharpLineModel)
SOFA_LINK_CLASS(TestDetection)
SOFA_LINK_CLASS(TriangularFEMFractureManager)
// constraint
SOFA_LINK_CLASS(BeamConstraint)
SOFA_LINK_CLASS(BilateralInteractionConstraint)
//SOFA_LINK_CLASS(LagrangianMultiplierAttachConstraint)
//SOFA_LINK_CLASS(LagrangianMultiplierContactConstraint)
//SOFA_LINK_CLASS(LagrangianMultiplierFixedConstraint)
SOFA_LINK_CLASS(SlidingConstraint)
// controller
SOFA_LINK_CLASS(JointSpringController)
// forcefield
SOFA_LINK_CLASS(HexahedronFEMForceFieldAndMass)
SOFA_LINK_CLASS(NonUniformHexahedronFEMForceFieldAndMass)
SOFA_LINK_CLASS(NonUniformHexahedronFEMForceFieldDensity)
SOFA_LINK_CLASS(TetrahedralBiquadraticSpringsForceField)
SOFA_LINK_CLASS(TetrahedralQuadraticSpringsForceField)
SOFA_LINK_CLASS(WashingMachineForceField)
SOFA_LINK_CLASS(Triangle2DFEMForceField)
SOFA_LINK_CLASS(TriangleBendingFEMForceField)
// material
SOFA_LINK_CLASS(HookeanMaterial)
SOFA_LINK_CLASS(PlasticMaterial)
// interactionforcefield
SOFA_LINK_CLASS(LagrangeMultiplierInteraction)
//mastersolver
SOFA_LINK_CLASS(MasterConstraintSolver)
//misc
SOFA_LINK_CLASS(EvalPointsDistance)
SOFA_LINK_CLASS(EvalSurfaceDistance)
//odesolver
SOFA_LINK_CLASS(ComplianceCGImplicitSolver)
SOFA_LINK_CLASS(ComplianceEuler)
SOFA_LINK_CLASS(BiCGStabImplicit)
//topology
SOFA_LINK_CLASS(FittedRegularGridTopology)

//simulation
//automatescheduler
SOFA_LINK_CLASS(ThreadSimulation)

#endif // SOFA_DEV

#ifdef SOFA_HAVE_SENSABLE

SOFA_LINK_CLASS(OmniDriver)
SOFA_LINK_CLASS(NullForceFeedback)
SOFA_LINK_CLASS(EnslavementForceFeedback)
SOFA_LINK_CLASS(LCPForceFeedback)
SOFA_LINK_CLASS(VectorSpringForceField)

#endif //SOFA_HAVE_SENSABLE


#ifdef SOFA_HAVE_ARBORIS
SOFA_LINK_CLASS(ArborisMapping)
#endif //SOFA_HAVE_ARBORIS
