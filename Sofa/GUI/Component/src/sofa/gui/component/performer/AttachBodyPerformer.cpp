/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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

#include <sofa/gui/component/performer/BaseAttachBodyPerformer.inl>
#include <sofa/gui/component/performer/AttachBodyPerformer.inl>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/Factory.inl>
#include <sofa/component/solidmechanics/spring/JointSpringForceField.inl>
#include <sofa/component/solidmechanics/spring/SpringForceField.inl>

using namespace sofa::core::objectmodel;

namespace sofa::gui::component::performer
{

template class SOFA_GUI_COMPONENT_API  AttachBodyPerformer<defaulttype::Vec2Types>;
template class SOFA_GUI_COMPONENT_API  AttachBodyPerformer<defaulttype::Vec3Types>;
template class SOFA_GUI_COMPONENT_API  AttachBodyPerformer<defaulttype::Rigid3Types>;

helper::Creator<InteractionPerformer::InteractionPerformerFactory, AttachBodyPerformer<defaulttype::Vec2Types> >  AttachBodyPerformerVec2dClass("AttachBody",true);
helper::Creator<InteractionPerformer::InteractionPerformerFactory, AttachBodyPerformer<defaulttype::Vec3Types> >  AttachBodyPerformerVec3dClass("AttachBody",true);
helper::Creator<InteractionPerformer::InteractionPerformerFactory, AttachBodyPerformer<defaulttype::Rigid3Types> >  AttachBodyPerformerRigid3dClass("AttachBody",true);

} // namespace sofa::gui::component::performer
