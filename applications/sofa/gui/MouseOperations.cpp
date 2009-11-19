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
#include <sofa/component/collision/PotentialInjectionPerformer.h>
#include <sofa/component/collision/RemovePrimitivePerformer.h>
#include <sofa/component/collision/InciseAlongPathPerformer.h>

namespace sofa
{

using namespace component::collision;

#ifdef WIN32
#ifndef SOFA_DOUBLE
helper::Creator<InteractionPerformer::InteractionPerformerFactory, AttachBodyPerformer<defaulttype::Vec3fTypes> >  AttachBodyPerformerVec3fClass("AttachBody",true);
helper::Creator<InteractionPerformer::InteractionPerformerFactory, FixParticlePerformer<defaulttype::Vec3fTypes> >  FixParticlePerformerVec3fClass("FixParticle",true);
#endif
#ifndef SOFA_FLOAT
helper::Creator<InteractionPerformer::InteractionPerformerFactory, AttachBodyPerformer<defaulttype::Vec3dTypes> >  AttachBodyPerformerVec3dClass("AttachBody",true);
helper::Creator<InteractionPerformer::InteractionPerformerFactory, FixParticlePerformer<defaulttype::Vec3dTypes> >  FixParticlePerformerVec3dClass("FixParticle",true);
#endif
helper::Creator<InteractionPerformer::InteractionPerformerFactory, RemovePrimitivePerformer >  RemovePrimitivePerformerClass("RemovePrimitive");
helper::Creator<InteractionPerformer::InteractionPerformerFactory, InciseAlongPathPerformer>  InciseAlongPathPerformerClass("InciseAlongPath");
helper::Creator<InteractionPerformer::InteractionPerformerFactory, PotentialInjectionPerformer> PotentialInjectionPerformerClass("SetActionPotential");
#endif

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
    std::cout << "AttachOperation::end()" << std::endl;

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
    int currentMethod = getIncisionMethod();

    if (currentMethod == 0) // incision clic by clic.
    {
        if (cpt == 0) // First clic => initialisation
        {
            performer=component::collision::InteractionPerformer::InteractionPerformerFactory::getInstance()->createObject("InciseAlongPath", pickHandle->getInteraction()->mouseInteractor);

            component::collision::InciseAlongPathPerformerconfiguration *performerConfiguration=dynamic_cast<component::collision::InciseAlongPathPerformerconfiguration*>(performer);
            performerConfiguration->setIncisionMethod(getIncisionMethod());

            pickHandle->getInteraction()->mouseInteractor->addInteractionPerformer(performer);
            performer->start();
            cpt++;
        }
        else // Second clic(s) only perform start() method
        {
            performer->start();
        }
    }
    else
    {
        performer=component::collision::InteractionPerformer::InteractionPerformerFactory::getInstance()->createObject("InciseAlongPath", pickHandle->getInteraction()->mouseInteractor);

        component::collision::InciseAlongPathPerformerconfiguration *performerConfiguration=dynamic_cast<component::collision::InciseAlongPathPerformerconfiguration*>(performer);
        performerConfiguration->setIncisionMethod(getIncisionMethod());

        pickHandle->getInteraction()->mouseInteractor->addInteractionPerformer(performer);
        performer->start();
    }
}


void InciseOperation::execution()
{
}

void InciseOperation::end()
{
}

void InciseOperation::endOperation()
{
    cpt = 0; //reinitialization
    pickHandle->getInteraction()->mouseInteractor->removeInteractionPerformer(performer);

}



//*******************************************************************************************
void InjectOperation::start()
{
    //Creation
    performer=component::collision::InteractionPerformer::InteractionPerformerFactory::getInstance()->createObject("SetActionPotential", pickHandle->getInteraction()->mouseInteractor);
    pickHandle->getInteraction()->mouseInteractor->addInteractionPerformer(performer);

    //Configuration
    component::collision::PotentialInjectionPerformerConfiguration *performerConfiguration=dynamic_cast<component::collision::PotentialInjectionPerformerConfiguration*>(performer);
    performerConfiguration->setPotentialValue(getPotentialValue());
    performerConfiguration->setStateTag(getStateTag());
    //Start
    performer->start();
}


void InjectOperation::execution()
{
//       performer->execute();
}

void InjectOperation::end()
{
    //   execution();
    pickHandle->getInteraction()->mouseInteractor->removeInteractionPerformer(performer);
}

}
}
