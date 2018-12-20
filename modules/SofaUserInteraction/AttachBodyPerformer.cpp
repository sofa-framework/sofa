/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#define SOFA_COMPONENT_COLLISION_ATTACHBODYPERFORMER_CPP

#include <SofaUserInteraction/AttachBodyPerformer.inl>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/Factory.inl>
#include <SofaRigid/JointSpringForceField.inl>
#include <SofaDeformable/SpringForceField.inl>
#include <SofaDeformable/StiffSpringForceField.inl>


using namespace sofa::component::interactionforcefield;
using namespace sofa::core::objectmodel;
namespace sofa
{

namespace component
{

namespace collision
{

template class SOFA_USER_INTERACTION_API  AttachBodyPerformer<defaulttype::Vec2Types>;
template class SOFA_USER_INTERACTION_API  AttachBodyPerformer<defaulttype::Vec3Types>;
template class SOFA_USER_INTERACTION_API  AttachBodyPerformer<defaulttype::Rigid3Types>;

static helper::Creator<InteractionPerformer::InteractionPerformerFactory, AttachBodyPerformer<defaulttype::Vec2Types> >  AttachBodyPerformerVec2dClass("AttachBody",true);
static helper::Creator<InteractionPerformer::InteractionPerformerFactory, AttachBodyPerformer<defaulttype::Vec3Types> >  AttachBodyPerformerVec3dClass("AttachBody",true);
static helper::Creator<InteractionPerformer::InteractionPerformerFactory, AttachBodyPerformer<defaulttype::Rigid3Types> >  AttachBodyPerformerRigid3dClass("AttachBody",true);

/*
template <>
bool AttachBodyPerformer<defaulttype::Rigid3Types>::start_partial(const BodyPicked& picked)
{
    core::behavior::MechanicalState<defaulttype::Rigid3Types>* mstateCollision=NULL;

    double restLength = picked.dist;
    mstateCollision = static_cast< core::behavior::MechanicalState<defaulttype::Rigid3Types>*  >(picked.mstate);

    if( !mstateCollision ) mstateCollision = static_cast< core::behavior::MechanicalState<defaulttype::Rigid3Types>* >( picked.body->getContext()->getMechanicalState() );

    if( !mstateCollision ) return false;

    m_forcefield = sofa::core::objectmodel::New< JointSpringForceField< defaulttype::Rigid3Types > >(dynamic_cast<MouseContainer*>(this->interactor->getMouseContainer()), mstateCollision);
    JointSpringForceField<defaulttype::Rigid3Types>* jointspringforcefield = static_cast<JointSpringForceField<defaulttype::Rigid3Types>*>(m_forcefield.get());
    sofa::component::interactionforcefield::JointSpring<defaulttype::Rigid3Types> spring(0,picked.indexCollisionElement);
    jointspringforcefield->setName("Spring-Mouse-Contact");


    spring.setInitLength(this->interactor->getMouseRayModel()->getRay(0).direction()*restLength);
    spring.setSoftStiffnessTranslation(stiffness);
    jointspringforcefield->addSpring(spring);
    jointspringforcefield->showFactorSize.setValue(showFactorSize);

    const core::objectmodel::TagSet &tags=mstateCollision->getTags();
    for (core::objectmodel::TagSet::const_iterator it=tags.begin(); it!=tags.end(); ++it)
        jointspringforcefield->addTag(*it);

    mstateCollision->getContext()->addObject(jointspringforcefield);

    return true;
}
*/



}
}
}
