/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/defaulttype/Vec3Types.h>
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

#ifndef SOFA_DOUBLE
template class SOFA_USER_INTERACTION_API  AttachBodyPerformer<defaulttype::Vec2fTypes>;
template class SOFA_USER_INTERACTION_API  AttachBodyPerformer<defaulttype::Vec3fTypes>;
template class SOFA_USER_INTERACTION_API  AttachBodyPerformer<defaulttype::Rigid3fTypes>;
#endif
#ifndef SOFA_FLOAT
template class SOFA_USER_INTERACTION_API  AttachBodyPerformer<defaulttype::Vec2dTypes>;
template class SOFA_USER_INTERACTION_API  AttachBodyPerformer<defaulttype::Vec3dTypes>;
template class SOFA_USER_INTERACTION_API  AttachBodyPerformer<defaulttype::Rigid3dTypes>;
#endif


#ifndef SOFA_DOUBLE
helper::Creator<InteractionPerformer::InteractionPerformerFactory, AttachBodyPerformer<defaulttype::Vec2fTypes> >  AttachBodyPerformerVec2fClass("AttachBody",true);
helper::Creator<InteractionPerformer::InteractionPerformerFactory, AttachBodyPerformer<defaulttype::Vec3fTypes> >  AttachBodyPerformerVec3fClass("AttachBody",true);
helper::Creator<InteractionPerformer::InteractionPerformerFactory, AttachBodyPerformer<defaulttype::Rigid3fTypes> >  AttachBodyPerformerRigid3fClass("AttachBody",true);
template <>
bool AttachBodyPerformer<defaulttype::Rigid3fTypes>::start_partial(const BodyPicked& picked)
{
    core::behavior::MechanicalState<defaulttype::Rigid3fTypes>* mstateCollision=NULL;

    double restLength = picked.dist;
    mstateCollision = static_cast< core::behavior::MechanicalState<defaulttype::Rigid3fTypes>*  >(picked.mstate);

    if( !mstateCollision ) mstateCollision = static_cast< core::behavior::MechanicalState<defaulttype::Rigid3fTypes>* >( picked.body->getContext()->getMechanicalState() );

    if( !mstateCollision ) return false;

    m_forcefield = sofa::core::objectmodel::New< JointSpringForceField< defaulttype::Rigid3fTypes > >(dynamic_cast<MouseContainer*>(this->interactor->getMouseContainer()), mstateCollision);
    sofa::component::interactionforcefield::JointSpring<defaulttype::Rigid3fTypes> spring(0,picked.indexCollisionElement);
    JointSpringForceField<defaulttype::Rigid3fTypes>* jointspringforcefield = static_cast<JointSpringForceField<defaulttype::Rigid3fTypes>*>(m_forcefield.get());

    jointspringforcefield->setName("Spring-Mouse-Contact");

    spring.setInitLength(this->interactor->getMouseRayModel()->getRay(0).direction()*restLength);
    spring.setSoftStiffnessTranslation((float)stiffness);
    jointspringforcefield->addSpring(spring);
    jointspringforcefield->showFactorSize.setValue((float)showFactorSize);
    const core::objectmodel::TagSet &tags=mstateCollision->getTags();
    for (core::objectmodel::TagSet::const_iterator it=tags.begin(); it!=tags.end(); ++it)
        jointspringforcefield->addTag(*it);
    mstateCollision->getContext()->addObject(jointspringforcefield);

    return true;
}

#endif
#ifndef SOFA_FLOAT
helper::Creator<InteractionPerformer::InteractionPerformerFactory, AttachBodyPerformer<defaulttype::Vec2dTypes> >  AttachBodyPerformerVec2dClass("AttachBody",true);
helper::Creator<InteractionPerformer::InteractionPerformerFactory, AttachBodyPerformer<defaulttype::Vec3dTypes> >  AttachBodyPerformerVec3dClass("AttachBody",true);
helper::Creator<InteractionPerformer::InteractionPerformerFactory, AttachBodyPerformer<defaulttype::Rigid3dTypes> >  AttachBodyPerformerRigid3dClass("AttachBody",true);

template <>
bool AttachBodyPerformer<defaulttype::Rigid3dTypes>::start_partial(const BodyPicked& picked)
{
    core::behavior::MechanicalState<defaulttype::Rigid3dTypes>* mstateCollision=NULL;

    double restLength = picked.dist;
    mstateCollision = static_cast< core::behavior::MechanicalState<defaulttype::Rigid3dTypes>*  >(picked.mstate);

    if( !mstateCollision ) mstateCollision = static_cast< core::behavior::MechanicalState<defaulttype::Rigid3dTypes>* >( picked.body->getContext()->getMechanicalState() );

    if( !mstateCollision ) return false;

    m_forcefield = sofa::core::objectmodel::New< JointSpringForceField< defaulttype::Rigid3dTypes > >(dynamic_cast<MouseContainer*>(this->interactor->getMouseContainer()), mstateCollision);
    JointSpringForceField<defaulttype::Rigid3dTypes>* jointspringforcefield = static_cast<JointSpringForceField<defaulttype::Rigid3dTypes>*>(m_forcefield.get());
    sofa::component::interactionforcefield::JointSpring<defaulttype::Rigid3dTypes> spring(0,picked.indexCollisionElement);
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
#endif


}
}
}
