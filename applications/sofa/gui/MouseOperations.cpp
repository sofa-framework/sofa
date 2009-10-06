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
#include <sofa/gui/MouseOperations.h>
#include <sofa/gui/PickHandler.h>
#include <sofa/component/collision/InteractionPerformer.h>

#include <sofa/component/collision/ComponentMouseInteraction.h>
#include <sofa/component/collision/AttachBodyPerformer.h>
#include <sofa/component/collision/FixParticlePerformer.h>
#ifdef SOFA_DEV
#include <sofa/component/collision/SculptBodyPerformer.h>
#endif
#include <sofa/component/collision/RemovePrimitivePerformer.h>
#include <sofa/component/collision/InciseAlongPathPerformer.h>

namespace sofa
{

using namespace component::collision;

namespace gui
{
//*******************************************************************************************
void AttachOperation::start()
{
    //Creation
    performer=component::collision::InteractionPerformer::InteractionPerformerFactory::getInstance()->createObject("AttachBody", pickHandle->getInteraction()->mouseInteractor);
    pickHandle->getInteraction()->mouseInteractor->addInteractionPerformer(performer);
    //Configuration
    component::collision::AttachBodyPerformerConfiguration *performerConfiguration=dynamic_cast<component::collision::AttachBodyPerformerConfiguration*>(performer);
    performerConfiguration->setStiffness(getStiffness());

    //Start
    performer->start();
}

void AttachOperation::execution()
{
    //do nothing
}

void AttachOperation::end()
{
    pickHandle->getInteraction()->mouseInteractor->removeInteractionPerformer(performer);
}

//*******************************************************************************************
void FixOperation::start()
{
    //Creation
    performer=component::collision::InteractionPerformer::InteractionPerformerFactory::getInstance()->createObject("FixParticle", pickHandle->getInteraction()->mouseInteractor);
    pickHandle->getInteraction()->mouseInteractor->addInteractionPerformer(performer);

    //Configuration
    component::collision::FixParticlePerformerConfiguration *performerConfiguration=dynamic_cast<component::collision::FixParticlePerformerConfiguration*>(performer);
    performerConfiguration->setStiffness(getStiffness());
    //Start
    performer->start();
}

void FixOperation::execution()
{
}

void FixOperation::end()
{
    //do nothing
}


//*******************************************************************************************
void RemoveOperation::start()
{
    performer=component::collision::InteractionPerformer::InteractionPerformerFactory::getInstance()->createObject("RemovePrimitive", pickHandle->getInteraction()->mouseInteractor);
    pickHandle->getInteraction()->mouseInteractor->addInteractionPerformer(performer);
}

void RemoveOperation::execution()
{
//       performer->execute();
}

void RemoveOperation::end()
{
    pickHandle->getInteraction()->mouseInteractor->removeInteractionPerformer(performer);
}



//*******************************************************************************************
void InciseOperation::start()
{
    performer=component::collision::InteractionPerformer::InteractionPerformerFactory::getInstance()->createObject("InciseAlongPath", pickHandle->getInteraction()->mouseInteractor);
    pickHandle->getInteraction()->mouseInteractor->addInteractionPerformer(performer);
    performer->start();
}


void InciseOperation::execution()
{
//       performer->execute();
}

void InciseOperation::end()
{
    execution();
    pickHandle->getInteraction()->mouseInteractor->removeInteractionPerformer(performer);
}


//*******************************************************************************************
void SculptOperation::start()
{
#ifdef SOFA_DEV
    if (performer)
    {
        component::collision::SculptBodyPerformerConfiguration *performerConfiguration=dynamic_cast<component::collision::SculptBodyPerformerConfiguration*>(performer);
        performerConfiguration->setForce(getForce()/5000);
    }
#endif
}

void SculptOperation::execution()
{
#ifdef SOFA_DEV
    performer->execute();
#endif
}

void SculptOperation::end()
{
#ifdef SOFA_DEV
    component::collision::SculptBodyPerformerConfiguration *performerConfiguration=dynamic_cast<component::collision::SculptBodyPerformerConfiguration*>(performer);
    performerConfiguration->setForce(0.0);
#endif
}

void SculptOperation::wait()
{
#ifdef SOFA_DEV
    if( performer==NULL || pickHandle->getInteraction()->mouseInteractor!= performer->interactor)
    {
        //Creation
        performer=component::collision::InteractionPerformer::InteractionPerformerFactory::getInstance()->createObject("SculptBody", pickHandle->getInteraction()->mouseInteractor);
        pickHandle->getInteraction()->mouseInteractor->addInteractionPerformer(performer);

        //Configuration
        component::collision::SculptBodyPerformerConfiguration *performerConfiguration=dynamic_cast<component::collision::SculptBodyPerformerConfiguration*>(performer);
        performerConfiguration->setScale(getScale());
    }
#endif
}
SculptOperation::~SculptOperation()
{
#ifdef SOFA_DEV
    pickHandle->getInteraction()->mouseInteractor->removeInteractionPerformer(performer);
#endif
}

}
}
