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

#ifdef WIN32
#ifndef SOFA_DOUBLE
#ifdef SOFA_DEV
helper::Creator<InteractionPerformer::InteractionPerformerFactory, CompliantAttachPerformer<defaulttype::Vec3fTypes> >  CompliantAttachPerformerVec3fClass("CompliantAttach",true);
#endif
#endif
#ifndef SOFA_FLOAT
#ifdef SOFA_DEV
helper::Creator<InteractionPerformer::InteractionPerformerFactory, CompliantAttachPerformer<defaulttype::Vec3dTypes> >  CompliantAttachPerformerVec3dClass("CompliantAttach",true);
#endif
#endif
#endif

namespace gui
{
//*******************************************************************************************
//void CompliantAttachOperation::start()
//{
//    //Creation
//    performer=component::collision::InteractionPerformer::InteractionPerformerFactory::getInstance()->createObject("CompliantAttach", pickHandle->getInteraction()->mouseInteractor.get());
//    pickHandle->getInteraction()->mouseInteractor->addInteractionPerformer(performer);
//    configurePerformer(performer);
//    //Start
//    performer->start();
//}

//void CompliantAttachOperation::execution()
//{
//    //do nothing
//}

//void CompliantAttachOperation::end()
//{
//    pickHandle->getInteraction()->mouseInteractor->removeInteractionPerformer(performer);
//    delete performer; performer=0;
//}

//void CompliantAttachOperation::endOperation()
//{
//    pickHandle->getInteraction()->mouseInteractor->removeInteractionPerformer(performer);
//}


//void CompliantAttachOperation::configurePerformer(sofa::component::collision::InteractionPerformer* p)
//{
//    Operation::configurePerformer(p);
//}


}// gui


namespace component
{

namespace collision
{

#ifndef SOFA_DOUBLE
template class SOFA_Compliant_gui_API  CompliantAttachPerformer<defaulttype::Vec3fTypes>;
template class SOFA_Compliant_gui_API  CompliantAttachPerformer<defaulttype::Rigid3fTypes>;
#endif
#ifndef SOFA_FLOAT
template class SOFA_Compliant_gui_API  CompliantAttachPerformer<defaulttype::Vec3dTypes>;
template class SOFA_Compliant_gui_API  CompliantAttachPerformer<defaulttype::Rigid3dTypes>;
#endif


#ifndef SOFA_DOUBLE
helper::Creator<InteractionPerformer::InteractionPerformerFactory, CompliantAttachPerformer<defaulttype::Vec3fTypes> >  CompliantAttachPerformerVec3fClass("CompliantAttach",true);
helper::Creator<InteractionPerformer::InteractionPerformerFactory, CompliantAttachPerformer<defaulttype::Rigid3fTypes> >  CompliantAttachPerformerRigid3fClass("CompliantAttach",true);

#endif
#ifndef SOFA_FLOAT
helper::Creator<InteractionPerformer::InteractionPerformerFactory, CompliantAttachPerformer<defaulttype::Vec3dTypes> >  CompliantAttachPerformerVec3dClass("CompliantAttach",true);
helper::Creator<InteractionPerformer::InteractionPerformerFactory, CompliantAttachPerformer<defaulttype::Rigid3dTypes> >  CompliantAttachPerformerRigid3dClass("CompliantAttach",true);
#endif
}
}
}
#endif
