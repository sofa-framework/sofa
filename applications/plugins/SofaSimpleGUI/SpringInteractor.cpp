#include <iostream>
using namespace std;
#include "SpringInteractor.h"
#include "PickedPoint.h"
#include <sofa/core/SofaLibrary.h>
#include <sofa/simulation/common/Simulation.h>

namespace sofa{
namespace simplegui{

SpringInteractor::SpringInteractor(const PickedPoint &picked, SReal stiffness)
    : Interactor(picked)
{
    // get the DOF of the picked object
    MechanicalObject3* pickedDof=dynamic_cast<MechanicalObject3*>(picked.state.get()); assert(pickedDof);

    // create DOF to represent the actuator
    _interactorDof = sofa::core::objectmodel::New<MechanicalObject3>();
    _interactionNode->addObject(_interactorDof);
    _interactorDof->setName("interactorDOF");
    _interactorDof->addTag(std::string("Interactor"));
    MechanicalObject3::WriteVecCoord xanchor = _interactorDof->writePositions();
    xanchor[0] = picked.point;
    FixedConstraint3::SPtr fixed= sofa::core::objectmodel::New<FixedConstraint3>(); // Makes it unmovable through forces
    _interactionNode->addObject(fixed);
    fixed->init();

    // create spring to drag the picked object
    _spring = sofa::core::objectmodel::New<StiffSpringForceField3>(_interactorDof.get(),pickedDof);
    _interactionNode->addObject(_spring);
    _spring->addSpring(0,picked.index,stiffness,0.1,0.);



//    cout << "SpringInteractor set spring to " << pickedDof->getName() << ", " << picked.index << endl;
}

Vec3 SpringInteractor::getPoint()
{
    MechanicalObject3::ReadVecCoord xanchor = _interactorDof->readPositions();
    return xanchor[0];
}

void SpringInteractor::setPoint( const Vec3& p )
{
    MechanicalObject3::WriteVecCoord xanchor = _interactorDof->writePositions();
    xanchor[0] = p;
}

void SpringInteractor::attach(SofaScene *scene)
{
    Inherited::attach(scene);
    _interactionNode->removeObject(_spring);
    Node* targetParent = dynamic_cast<Node*>(_spring->getMState2()->getContext());
    targetParent->addObject(_spring);
}

void SpringInteractor::detach()
{
    Inherited::detach();
    Node* parent = dynamic_cast<Node*>(_spring->getMState2()->getContext());
    parent->removeObject(_spring);
    _interactionNode->addObject(_spring);
}

}//newgui
}//sofa
