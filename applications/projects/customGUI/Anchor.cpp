#include "Anchor.h"
#include "GlPickedPoint.h"
#include <sofa/simulation/common/Simulation.h>
#include <sofa/simulation/common/DeleteVisitor.h>

using namespace sofa;

Anchor::Anchor(GlPickedPoint *picked)
{
    MechanicalObject3* pickedDof=dynamic_cast<MechanicalObject3*>(picked->state.get()); assert(pickedDof);

    // use a spring for interaction
    interactionNode = sofa::simulation::getSimulation()->createNewNode("picked point interaction node");
    anchorDof = New<MechanicalObject3>();
    interactionNode->addObject(anchorDof);
    StiffSpringForceField3::SPtr spring = New<StiffSpringForceField3>(anchorDof.get(),pickedDof);
    interactionNode->addObject(spring);
    spring->addSpring(0,picked->index,100,0.1,0.);

}

Anchor::~Anchor()
{
    interactionNode->execute<simulation::DeleteVisitor>(core::ExecParams::defaultInstance());
}

void Anchor::attach(Node::SPtr parent)
{
    parent->addChild(interactionNode);
}

void Anchor::detach()
{
    interactionNode->detachFromGraph();
}

