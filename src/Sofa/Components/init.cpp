#include "Common/config.h"
#include <iostream>

namespace Sofa
{

namespace Components
{

void init()
{
    std::cout << "Sofa Components Initialized"<<std::endl;
}

} // namespace Components

} // namespace SOFA

////////// BEGIN CLASS LIST //////////
SOFA_LINK_CLASS(BarycentricMapping)
SOFA_LINK_CLASS(BehaviorModel)
SOFA_LINK_CLASS(BruteForce)
SOFA_LINK_CLASS(CGImplicit)
SOFA_LINK_CLASS(CollisionDetection)
SOFA_LINK_CLASS(CollisionGroup)
SOFA_LINK_CLASS(CollisionGroupManagerSofa)
SOFA_LINK_CLASS(CollisionModel)
SOFA_LINK_CLASS(CollisionPipeline)
SOFA_LINK_CLASS(Constraint)
SOFA_LINK_CLASS(Contact)
SOFA_LINK_CLASS(ContactManagerSofa)
SOFA_LINK_CLASS(DiagonalMass)
SOFA_LINK_CLASS(DynamicModel)
SOFA_LINK_CLASS(Euler)
SOFA_LINK_CLASS(FixedConstraint)
SOFA_LINK_CLASS(ForceField)
SOFA_LINK_CLASS(GNode)
SOFA_LINK_CLASS(Gravity)
SOFA_LINK_CLASS(GridTopology)
SOFA_LINK_CLASS(IdentityMapping)
SOFA_LINK_CLASS(SurfaceIdentityMapping)
SOFA_LINK_CLASS(ImageBMP)
SOFA_LINK_CLASS(InteractionForceField)
SOFA_LINK_CLASS(Intersection)
SOFA_LINK_CLASS(Mapping)
SOFA_LINK_CLASS(Mass)
SOFA_LINK_CLASS(MassObject)
SOFA_LINK_CLASS(MechanicalGroup)
SOFA_LINK_CLASS(MechanicalModel)
SOFA_LINK_CLASS(MeshOBJ)
SOFA_LINK_CLASS(MeshTopology)
SOFA_LINK_CLASS(Node)
SOFA_LINK_CLASS(OglModel)
SOFA_LINK_CLASS(PenalityContact)
SOFA_LINK_CLASS(PipelineSofa)
SOFA_LINK_CLASS(PlaneForceField)
SOFA_LINK_CLASS(Ray)
SOFA_LINK_CLASS(RayContact)
SOFA_LINK_CLASS(RegularGridSpringForceField)
SOFA_LINK_CLASS(RegularGridTopology)
SOFA_LINK_CLASS(RepulsiveSpringForceField)
SOFA_LINK_CLASS(RigidMapping)
SOFA_LINK_CLASS(RigidObject)
SOFA_LINK_CLASS(Rotation)
SOFA_LINK_CLASS(RungeKutta4)
SOFA_LINK_CLASS(Scene)
SOFA_LINK_CLASS(SceneNode)
SOFA_LINK_CLASS(Solver)
SOFA_LINK_CLASS(Sphere)
SOFA_LINK_CLASS(SpringForceField)
SOFA_LINK_CLASS(StaticSolver)
SOFA_LINK_CLASS(StiffSpringForceField)
SOFA_LINK_CLASS(TensorForceField)
SOFA_LINK_CLASS(Topology)
SOFA_LINK_CLASS(Trans)
SOFA_LINK_CLASS(UniformMass)
SOFA_LINK_CLASS(VisualModel)
SOFA_LINK_CLASS(VoxelGrid)
SOFA_LINK_CLASS(TetrahedronFEMForceField)
