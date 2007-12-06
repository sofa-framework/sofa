#include <sofa/helper/system/config.h>
#include <sofa/core/ObjectFactory.h>
#include <iostream>

namespace sofa
{

namespace simulation
{

namespace tree
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

} // namespace tree

} // namespace simulation

} // namespace sofa

////////// BEGIN CLASS LIST //////////
SOFA_LINK_CLASS(ArticulatedHierarchyContainer)
SOFA_LINK_CLASS(ArticulatedSystemMapping)
SOFA_LINK_CLASS(Articulation)
SOFA_LINK_CLASS(ArticulationCenter)
SOFA_LINK_CLASS(Attribute)
SOFA_LINK_CLASS(BarycentricLagrangianMultiplierContact)
SOFA_LINK_CLASS(BarycentricMapping)
SOFA_LINK_CLASS(BarycentricPenalityContact)
SOFA_LINK_CLASS(BeamFEMForceField)
SOFA_LINK_CLASS(BeamLinearMapping)
SOFA_LINK_CLASS(BruteForce)
SOFA_LINK_CLASS(CGImplicit)
SOFA_LINK_CLASS(ComplianceCGImplicitSolver)
SOFA_LINK_CLASS(CuttingManager)
SOFA_LINK_CLASS(DefaultCollisionGroupManager)
SOFA_LINK_CLASS(DefaultContactManager)
SOFA_LINK_CLASS(DefaultMasterSolver)
SOFA_LINK_CLASS(ContinuousIntersection)
SOFA_LINK_CLASS(ComplianceEuler)
SOFA_LINK_CLASS(ComplianceArticulatedSystemSolver)
SOFA_LINK_CLASS(ConstantForceField)
SOFA_LINK_CLASS(CoordinateSystem)
SOFA_LINK_CLASS(Cube)
SOFA_LINK_CLASS(CubeTopology)
SOFA_LINK_CLASS(DiagonalMass)
SOFA_LINK_CLASS(DiscreteIntersection)
SOFA_LINK_CLASS(DistanceGridCollisionModel)
SOFA_LINK_CLASS(EdgePressureForceField)
SOFA_LINK_CLASS(EdgeRemoveContact)
SOFA_LINK_CLASS(EdgeSetTopology)
SOFA_LINK_CLASS(EllipsoidForceField)
SOFA_LINK_CLASS(Euler)
SOFA_LINK_CLASS(ExternalForceField)
SOFA_LINK_CLASS(FixedConstraint)
SOFA_LINK_CLASS(FixedPlaneConstraint)
SOFA_LINK_CLASS(Fluid2D)
SOFA_LINK_CLASS(Fluid3D)
SOFA_LINK_CLASS(FrictionContact)
SOFA_LINK_CLASS(GNode)
SOFA_LINK_CLASS(Gravity)
SOFA_LINK_CLASS(GridTopology)
SOFA_LINK_CLASS(IdentityMapping)
SOFA_LINK_CLASS(ImageBMP)
SOFA_LINK_CLASS(ImagePNG)
SOFA_LINK_CLASS(ImplicitSurfaceMapping)
SOFA_LINK_CLASS(JointSpringForceField)
SOFA_LINK_CLASS(LagrangianMultiplierAttachConstraint)
SOFA_LINK_CLASS(LagrangianMultiplierContactConstraint)
SOFA_LINK_CLASS(LagrangianMultiplierFixedConstraint)
SOFA_LINK_CLASS(LaparoscopicRigidMapping)
SOFA_LINK_CLASS(LennardJonesForceField)
SOFA_LINK_CLASS(Line)
SOFA_LINK_CLASS(LineSetSkinningMapping)
SOFA_LINK_CLASS(MappedObject)
SOFA_LINK_CLASS(MasterContactSolver)
SOFA_LINK_CLASS(MechanicalObject)
SOFA_LINK_CLASS(MeshOBJ)
SOFA_LINK_CLASS(MeshSpringForceField)
SOFA_LINK_CLASS(MeshTopology)
SOFA_LINK_CLASS(MeshTrian)
SOFA_LINK_CLASS(MinProximityIntersection)
#ifdef SOFA_HAVE_MKL
SOFA_LINK_CLASS(MKLSolver)
#endif
SOFA_LINK_CLASS(NewMatSolver)
SOFA_LINK_CLASS(NewProximityIntersection)
SOFA_LINK_CLASS(Node)
SOFA_LINK_CLASS(Object)
SOFA_LINK_CLASS(OglModel)
SOFA_LINK_CLASS(OscillatorConstraint)
SOFA_LINK_CLASS(PenalityContactForceField)
SOFA_LINK_CLASS(DefaultPipeline)
SOFA_LINK_CLASS(PlaneForceField)
SOFA_LINK_CLASS(Point)
SOFA_LINK_CLASS(PointSetTopology)
SOFA_LINK_CLASS(ProximityIntersection)
SOFA_LINK_CLASS(Ray)
SOFA_LINK_CLASS(RayContact)
SOFA_LINK_CLASS(RegularGridSpringForceField)
SOFA_LINK_CLASS(RegularGridTopology)
SOFA_LINK_CLASS(RepulsiveSpringForceField)
SOFA_LINK_CLASS(RigidMapping)
SOFA_LINK_CLASS(RigidRigidMapping)
SOFA_LINK_CLASS(RungeKutta4)
SOFA_LINK_CLASS(BoxConstraint)
//SOFA_LINK_CLASS(SparseGridSpringForceField)
SOFA_LINK_CLASS(SharpLineModel)
SOFA_LINK_CLASS(SkinningMapping)
SOFA_LINK_CLASS(SparseGridTopology)
SOFA_LINK_CLASS(HexahedronFEMForceField)
SOFA_LINK_CLASS(HexahedronFEMForceFieldAndMass)
SOFA_LINK_CLASS(Sphere)
SOFA_LINK_CLASS(SphereForceField)
SOFA_LINK_CLASS(ConicalForceField)
SOFA_LINK_CLASS(SPHFluidForceField)
SOFA_LINK_CLASS(SPHFluidSurfaceMapping)
SOFA_LINK_CLASS(SpringForceField)
//SOFA_LINK_CLASS(SpringEdgeDataForceField)
SOFA_LINK_CLASS(StaticSolver)
SOFA_LINK_CLASS(StiffSpringForceField)
SOFA_LINK_CLASS(SurfaceIdentityMapping)
SOFA_LINK_CLASS(SubsetMapping)
//SOFA_LINK_CLASS(TensorForceField)
SOFA_LINK_CLASS(TetrahedronFEMForceField)
SOFA_LINK_CLASS(TetrahedralTensorMassForceField)
SOFA_LINK_CLASS(TetrahedralBiquadraticSpringsForceField)
SOFA_LINK_CLASS(TetrahedralCorotationalFEMForceField)
SOFA_LINK_CLASS(TetrahedralQuadraticSpringsForceField)
SOFA_LINK_CLASS(ThreadSimulation)
SOFA_LINK_CLASS(Triangle)
SOFA_LINK_CLASS(TriangleBendingSprings)
SOFA_LINK_CLASS(TriangularBendingSprings)
SOFA_LINK_CLASS(TrianglePressureForceField)
SOFA_LINK_CLASS(TriangleFEMForceField)
SOFA_LINK_CLASS(TriangularFEMForceField)
SOFA_LINK_CLASS(TriangularQuadraticSpringsForceField)
SOFA_LINK_CLASS(TriangularBiquadraticSpringsForceField)
SOFA_LINK_CLASS(TriangularTensorMassForceField)
SOFA_LINK_CLASS(FittedRegularGridTopology)
SOFA_LINK_CLASS(UniformMass)
SOFA_LINK_CLASS(Data)
SOFA_LINK_CLASS(TestDetection)
SOFA_LINK_CLASS(SphereTreeModel)
SOFA_LINK_CLASS(WashingMachineForceField)
SOFA_LINK_CLASS(ReadState)
SOFA_LINK_CLASS(WriteState)
SOFA_LINK_CLASS(QuadBendingSprings)
SOFA_LINK_CLASS(OglShader)
SOFA_LINK_CLASS(DirectionalLight)
SOFA_LINK_CLASS(PositionalLight)
SOFA_LINK_CLASS(SpotLight)

