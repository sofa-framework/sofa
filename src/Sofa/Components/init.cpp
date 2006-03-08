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
SOFA_LINK_CLASS(CollisionModel)
SOFA_LINK_CLASS(Contact)
SOFA_LINK_CLASS(CollisionDetection)
SOFA_LINK_CLASS(ForceField)
SOFA_LINK_CLASS(SceneNode)
SOFA_LINK_CLASS(BehaviorModel)
SOFA_LINK_CLASS(DynamicModel)
SOFA_LINK_CLASS(Mapping)
SOFA_LINK_CLASS(MechanicalGroup)
SOFA_LINK_CLASS(MechanicalModel)
SOFA_LINK_CLASS(VisualModel)
SOFA_LINK_CLASS(InteractionForceField)
SOFA_LINK_CLASS(Solver)
SOFA_LINK_CLASS(CollisionGroup)
SOFA_LINK_CLASS(CollisionPipeline)
SOFA_LINK_CLASS(Trans)
SOFA_LINK_CLASS(OglModel)
SOFA_LINK_CLASS(ContactManagerSofa)
SOFA_LINK_CLASS(BruteForce)
SOFA_LINK_CLASS(VoxelGrid)
SOFA_LINK_CLASS(PipelineSofa)
SOFA_LINK_CLASS(CollisionGroupManagerSofa)
SOFA_LINK_CLASS(Sphere)
SOFA_LINK_CLASS(StiffSpringForceField)
SOFA_LINK_CLASS(SpringForceField)
SOFA_LINK_CLASS(MassObject)
SOFA_LINK_CLASS(Euler)
SOFA_LINK_CLASS(RungeKutta4)
SOFA_LINK_CLASS(CGImplicit)
SOFA_LINK_CLASS(MeshOBJ)
SOFA_LINK_CLASS(PenalityContact)
SOFA_LINK_CLASS(IdentityMapping)
SOFA_LINK_CLASS(RigidMapping)
SOFA_LINK_CLASS(ImageBMP)
SOFA_LINK_CLASS(RigidObject)
SOFA_LINK_CLASS(Scene)
SOFA_LINK_CLASS(RepulsiveSpringForceField)
