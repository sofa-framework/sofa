/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "MouseOperations.h"
#include <sofa/gui/PickHandler.h>
#include <sofa/component/collision/ComponentMouseInteraction.h>
#include <plugins/PhysicsBasedInteractiveModeler/pim/SculptBodyPerformer.h>

namespace plugins
{
namespace pim
{
namespace gui
{

using namespace sofa::defaulttype;

#ifdef WIN32
#ifndef SOFA_DOUBLE
helper::Creator<InteractionPerformer::InteractionPerformerFactory, SculptBodyPerformer<defaulttype::Vec3fTypes> >  SculptBodyPerformerVec3fClass("SculptBody",true);
#endif
#ifndef SOFA_FLOAT
helper::Creator<InteractionPerformer::InteractionPerformerFactory, SculptBodyPerformer<defaulttype::Vec3dTypes> >  SculptBodyPerformerVec3dClass("SculptBody",true);
#endif
#endif

void SculptOperation::start()
{
    if (performer == NULL) return;

    SculptBodyPerformerConfiguration *performerConfiguration=dynamic_cast<SculptBodyPerformerConfiguration*>(performer);
    performerConfiguration->setCheckedFix(isCheckedFix());
    performerConfiguration->setForce(getForce()/500);
}

void SculptOperation::end()
{
    if (performer == NULL) return;

    SculptBodyPerformerConfiguration *performerConfiguration=dynamic_cast<SculptBodyPerformerConfiguration*>(performer);
    performerConfiguration->setForce(0.0);
    performerConfiguration->setCheckedFix(false);
    SculptBodyPerformer<Vec3Types>* sculptPerformer=dynamic_cast<SculptBodyPerformer<Vec3Types>*>(performer);
    sculptPerformer->end();
    if (isAnimated())
    {
        sculptPerformer->animate(true);
    }
}

void SculptOperation::wait()
{
    if (performer==NULL && pickHandle->getInteraction()->mouseInteractor->getBodyPicked().body != NULL)
    {
        performer=InteractionPerformer::InteractionPerformerFactory::getInstance()->createObject("SculptBody", pickHandle->getInteraction()->mouseInteractor);
        pickHandle->getInteraction()->mouseInteractor->addInteractionPerformer(performer);
        SculptBodyPerformerConfiguration *performerConfiguration=dynamic_cast<SculptBodyPerformerConfiguration*>(performer);
        performerConfiguration->setScale(getScale());
        performerConfiguration->setForce(0.0);
        performerConfiguration->setCheckedFix(false);
    }
}

SculptOperation::~SculptOperation()
{
    if (performer != NULL)
        pickHandle->getInteraction()->mouseInteractor->removeInteractionPerformer(performer);
}

} // namespace gui
} // namespace pim
} // namespace plugins
