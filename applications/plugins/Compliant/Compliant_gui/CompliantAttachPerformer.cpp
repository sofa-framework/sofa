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
#ifndef SOFA_COMPONENT_COLLISION_CompliantAttachPerformer_CPP
#define SOFA_COMPONENT_COLLISION_CompliantAttachPerformer_CPP

#include "CompliantAttachPerformer.inl"
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/helper/Factory.inl>
#include <sofa/gui/PickHandler.h>
#include <SofaUserInteraction/ComponentMouseInteraction.h>

namespace sofa
{
using namespace component::collision;

namespace component
{

namespace collision
{

template class SOFA_Compliant_gui_API  CompliantAttachPerformer<defaulttype::Vec3Types>;
template class SOFA_Compliant_gui_API  CompliantAttachPerformer<defaulttype::Rigid3Types>;

helper::Creator<InteractionPerformer::InteractionPerformerFactory, CompliantAttachPerformer<defaulttype::Vec3Types> >  CompliantAttachPerformerVec3dClass("CompliantAttach",true);
helper::Creator<InteractionPerformer::InteractionPerformerFactory, CompliantAttachPerformer<defaulttype::Rigid3Types> >  CompliantAttachPerformerRigid3Class("CompliantAttach",true);

}
}
}
#endif
