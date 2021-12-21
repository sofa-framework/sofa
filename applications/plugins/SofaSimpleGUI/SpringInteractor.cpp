#include <iostream>
#include "SpringInteractor.h"

#include <boost/multi_index_container.hpp>

#include "PickedPoint.h"
#include <sofa/core/SofaLibrary.h>
#include <sofa/simulation/Simulation.h>
#include <SofaBoundaryCondition/FixedConstraint.h>

namespace sofa{
namespace simplegui{
typedef sofa::component::projectiveconstraintset::FixedConstraint<sofa::defaulttype::Vec3Types> FixedConstraint3;

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
    auto createdComponents =
    sofa::component::interactionforcefield::CreateSpringBetweenObjects<StiffSpringForceField3 >(
        _interactionNode.get(),
        _interactorDof.get(),
        pickedDof,
        {component::interactionforcefield::LinearSpring<StiffSpringForceField3::Real>{0, static_cast<sofa::Index>(picked.index), stiffness, 0.1, 0.} }
    );
    m_node = std::get<0>(createdComponents);
    m_mstate = std::get<1>(createdComponents);
    m_subsetMapping = std::get<2>(createdComponents);
    _spring = std::get<3>(createdComponents);

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
    _interactionNode->removeChild(m_node);
    Node* targetParent = dynamic_cast<Node*>(m_subsetMapping->getMechFrom()[1]->getContext());
    targetParent->addChild(m_node);
}

void SpringInteractor::detach()
{
    Inherited::detach();
    Node* parent = dynamic_cast<Node*>(m_subsetMapping->getMechFrom()[1]->getContext());
    parent->removeChild(m_node);
    _interactionNode->addChild(m_node);
}

}//newgui
}//sofa
