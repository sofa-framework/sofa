#include <iostream>
using namespace std;
#include "SpringInteractor.h"
#include "PickedPoint.h"
#include <sofa/simulation/common/Simulation.h>

namespace sofa{
namespace newgui{

SpringInteractor::SpringInteractor(const PickedPoint &picked)
    : Interactor(picked)
{
    // get the DOF of the picked object
    MechanicalObject3* pickedDof=dynamic_cast<MechanicalObject3*>(picked.state.get()); assert(pickedDof);

    // create DOF to represent the actuator
    anchorDof = New<MechanicalObject3>();
    interactionNode->addObject(anchorDof);
    MechanicalObject3::WriteVecCoord xanchor = anchorDof->writePositions();
    xanchor[0] = picked.point;

    // create spring to drag the picked object
    StiffSpringForceField3::SPtr spring = New<StiffSpringForceField3>(anchorDof.get(),pickedDof);
    interactionNode->addObject(spring);
    spring->addSpring(0,picked.index,100,0.1,0.);

//    cout << "DragAnchor::DragAnchor, set spring to " << pickedDof->getName() << ", " << picked.index << endl;
}

Vec3 SpringInteractor::getPoint()
{
    MechanicalObject3::ReadVecCoord xanchor = anchorDof->readPositions();
    return xanchor[0];
}

void SpringInteractor::setPoint( const Vec3& p )
{
    MechanicalObject3::WriteVecCoord xanchor = anchorDof->writePositions();
    xanchor[0] = p;
//    cout<<"DragAnchor::move to " << p << endl;
}

}
}
