#include <iostream>
using namespace std;
#include "SpringInteractor.h"
#include "PickedPoint.h"
#include <sofa/simulation/common/Simulation.h>

namespace sofa{
namespace newgui{

SpringInteractor::SpringInteractor(const PickedPoint &picked, SReal stiffness)
    : Interactor(picked)
{
    // get the DOF of the picked object
    MechanicalObject3* pickedDof=dynamic_cast<MechanicalObject3*>(picked.state.get()); assert(pickedDof);

    // create DOF to represent the actuator
    interactorDof = New<MechanicalObject3>();
    _interactionNode->addObject(interactorDof);
    interactorDof->setName("interactorDOF");
    MechanicalObject3::WriteVecCoord xanchor = interactorDof->writePositions();
    xanchor[0] = picked.point;

    // create spring to drag the picked object
    spring = New<StiffSpringForceField3>(interactorDof.get(),pickedDof);
    _interactionNode->addObject(spring);
    spring->addSpring(0,picked.index,stiffness,0.1,0.);

//    cout << "SpringInteractor set spring to " << pickedDof->getName() << ", " << picked.index << endl;
}

Vec3 SpringInteractor::getPoint()
{
    MechanicalObject3::ReadVecCoord xanchor = interactorDof->readPositions();
    return xanchor[0];
}

void SpringInteractor::setPoint( const Vec3& p )
{
    MechanicalObject3::WriteVecCoord xanchor = interactorDof->writePositions();
    xanchor[0] = p;
}

void SpringInteractor::attach(SofaScene *scene)
{
    Inherited::attach(scene);
    _interactionNode->removeObject(spring);
    Node* targetParent = dynamic_cast<Node*>(spring->getMState2()->getContext());
    targetParent->addObject(spring);
}

void SpringInteractor::detach()
{
    Inherited::detach();
    Node* parent = dynamic_cast<Node*>(spring->getMState2()->getContext());
    parent->removeObject(spring);
    _interactionNode->addObject(spring);
}

}//newgui
}//sofa
