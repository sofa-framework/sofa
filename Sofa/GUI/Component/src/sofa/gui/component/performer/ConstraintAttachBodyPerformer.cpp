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
#define SOFA_COMPONENT_COLLISION_CONSTRAINTATTACHBODYPERFORMER_CPP

#include <sofa/gui/component/performer/BaseAttachBodyPerformer.inl>
#include <sofa/gui/component/performer/ConstraintAttachBodyPerformer.inl>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/Factory.inl>


using namespace sofa::core::objectmodel;
namespace sofa::gui::component::performer
{

template class SOFA_GUI_COMPONENT_API ConstraintAttachBodyPerformer<defaulttype::Vec3Types>;
helper::Creator<InteractionPerformer::InteractionPerformerFactory, ConstraintAttachBodyPerformer<defaulttype::Vec3Types> >  ConstraintAttachBodyPerformerVec3dClass("ConstraintAttachBody",true);

template class SOFA_GUI_COMPONENT_API ConstraintAttachBodyPerformer<defaulttype::RigidTypes>;
helper::Creator<InteractionPerformer::InteractionPerformerFactory, ConstraintAttachBodyPerformer<defaulttype::RigidTypes> >  ConstraintAttachBodyPerformerRigidClass("ConstraintAttachBody",true);

} // namespace sofa::gui::component::performer
