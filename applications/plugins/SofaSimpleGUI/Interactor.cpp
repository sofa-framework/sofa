#include "Interactor.h"
#include "PickedPoint.h"
#include "SofaScene.h"
#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/DeleteVisitor.h>

using namespace sofa;

namespace sofa{
namespace simplegui{

Interactor::Interactor(const PickedPoint& picked)
    : _pickedPoint( picked )
{
    _interactionNode = sofa::simulation::getSimulation()->createNewNode("picked point interaction node");
}

Interactor::~Interactor()
{
    _interactionNode->execute<simulation::DeleteVisitor>(core::ExecParams::defaultInstance());
}

void Interactor::attach(SofaScene *scene)
{
    scene->insertInteractor(this);
}

void Interactor::detach()
{
    _interactionNode->detachFromGraph();
}

}
}
