#include "Interactor.h"
#include "PickedPoint.h"
#include "SofaScene.h"
#include <sofa/simulation/common/Simulation.h>
#include <sofa/simulation/common/DeleteVisitor.h>

using namespace sofa;

namespace sofa{
namespace newgui{

Interactor::Interactor(const PickedPoint& /*picked*/)
{
    interactionNode = sofa::simulation::getSimulation()->createNewNode("picked point interaction node");
}

Interactor::~Interactor()
{
    interactionNode->execute<simulation::DeleteVisitor>(core::ExecParams::defaultInstance());
}

void Interactor::attach(SofaScene *scene)
{
    scene->insertInteractor(this);
}

void Interactor::detach()
{
    interactionNode->detachFromGraph();
}

}
}
