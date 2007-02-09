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
        std::cout << "Sofa components initialized."<<std::endl;
        sofa::core::ObjectFactory::getInstance()->dump();
        std::ofstream ofile("sofa-classes.html");
        ofile << "<html><body>\n";
        sofa::core::ObjectFactory::getInstance()->dumpHTML(ofile);
        ofile << "</body></html>\n";
        first = false;
    }
}

} // namespace tree

} // namespace simulation

} // namespace sofa

////////// BEGIN CLASS LIST //////////
SOFA_LINK_CLASS(BarycentricLagrangianMultiplierContact)
SOFA_LINK_CLASS(BarycentricMapping)
SOFA_LINK_CLASS(BarycentricPenalityContact)
SOFA_LINK_CLASS(BruteForce)
SOFA_LINK_CLASS(CGImplicit)
SOFA_LINK_CLASS(MatrixStatic)
SOFA_LINK_CLASS(DefaultCollisionGroupManager)
SOFA_LINK_CLASS(DefaultContactManager)
SOFA_LINK_CLASS(ContinuousIntersection)
SOFA_LINK_CLASS(CoordinateSystem)
SOFA_LINK_CLASS(Cube)
SOFA_LINK_CLASS(DiagonalMass)
SOFA_LINK_CLASS(DiscreteIntersection)
SOFA_LINK_CLASS(EdgePressureForceField)
SOFA_LINK_CLASS(EdgeSetTopology)
SOFA_LINK_CLASS(Euler)
SOFA_LINK_CLASS(ExternalForceField)
SOFA_LINK_CLASS(FixedConstraint)
SOFA_LINK_CLASS(FixedPlaneConstraint)
SOFA_LINK_CLASS(GNode)
SOFA_LINK_CLASS(Gravity)
SOFA_LINK_CLASS(GridTopology)
SOFA_LINK_CLASS(IdentityMapping)
SOFA_LINK_CLASS(ImageBMP)
SOFA_LINK_CLASS(ImagePNG)
SOFA_LINK_CLASS(ImplicitSurfaceMapping)
SOFA_LINK_CLASS(InteractionConstraintImpl)
SOFA_LINK_CLASS(LagrangianMultiplierAttachConstraint)
SOFA_LINK_CLASS(LagrangianMultiplierContactConstraint)
SOFA_LINK_CLASS(LagrangianMultiplierFixedConstraint)
SOFA_LINK_CLASS(LaparoscopicRigidMapping)
SOFA_LINK_CLASS(LennardJonesForceField)
SOFA_LINK_CLASS(Line)
SOFA_LINK_CLASS(MechanicalObject)
SOFA_LINK_CLASS(MeshOBJ)
SOFA_LINK_CLASS(MeshSpringForceField)
SOFA_LINK_CLASS(MeshTopology)
SOFA_LINK_CLASS(MeshTrian)
SOFA_LINK_CLASS(MinProximityIntersection)
//SOFA_LINK_CLASS(MultiResSparseGridTopology)
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
SOFA_LINK_CLASS(RungeKutta4)
SOFA_LINK_CLASS(BoxConstraint)
//SOFA_LINK_CLASS(SparseGridSpringForceField)
SOFA_LINK_CLASS(Sphere)
SOFA_LINK_CLASS(SPHFluidForceField)
SOFA_LINK_CLASS(SPHFluidSurfaceMapping)
SOFA_LINK_CLASS(SpringForceField)
//SOFA_LINK_CLASS(SpringEdgeDataForceField)
SOFA_LINK_CLASS(StaticSolver)
SOFA_LINK_CLASS(StiffSpringForceField)
SOFA_LINK_CLASS(SurfaceIdentityMapping)
//SOFA_LINK_CLASS(TensorForceField)
SOFA_LINK_CLASS(TetrahedronFEMForceField)
SOFA_LINK_CLASS(ThreadSimulation)
SOFA_LINK_CLASS(Triangle)
SOFA_LINK_CLASS(TriangleBendingSprings)
SOFA_LINK_CLASS(TriangleFEMForceField)
SOFA_LINK_CLASS(FittedRegularGridTopology)
SOFA_LINK_CLASS(UniformMass)
SOFA_LINK_CLASS(TestDetection)
SOFA_LINK_CLASS(SphereTreeModel)
//SOFA_LINK_CLASS(VoxelGrid)
SOFA_LINK_CLASS(WashingMachineForceField)
