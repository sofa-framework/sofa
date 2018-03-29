/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/gui/MouseOperations.h>
#include <sofa/gui/PickHandler.h>
#include <SofaUserInteraction/InteractionPerformer.h>

#include <SofaUserInteraction/ComponentMouseInteraction.h>
#include <SofaUserInteraction/AttachBodyPerformer.h>
#include <SofaUserInteraction/FixParticlePerformer.h>
#include <SofaUserInteraction/RemovePrimitivePerformer.h>
#include <SofaUserInteraction/InciseAlongPathPerformer.h>
#include <SofaUserInteraction/AddRecordedCameraPerformer.h>
#include <SofaUserInteraction/StartNavigationPerformer.h>
#include <SofaUserInteraction/SuturePointPerformer.h>
#ifdef SOFA_HAVE_ARPLUGIN
#include "./../../../applications-dev/plugins/ARPlugin/ARPSAttachPerformer.h"
#endif

namespace sofa
{

using namespace component::collision;

#ifdef WIN32
#ifndef SOFA_DOUBLE
helper::Creator<InteractionPerformer::InteractionPerformerFactory, AttachBodyPerformer<defaulttype::Vec3fTypes> >  AttachBodyPerformerVec3fClass("AttachBody",true);
#ifdef SOFA_DEV
helper::Creator<InteractionPerformer::InteractionPerformerFactory, AddFramePerformer<defaulttype::Vec3fTypes> >  AddFramePerformerVec3fClass("AddFrame",true);
#ifdef SOFA_HAVE_ARPLUGIN
helper::Creator<InteractionPerformer::InteractionPerformerFactory, AdaptativeAttachPerformer<defaulttype::Vec3fTypes> >  AdaptativeAttachPerformerVec3fClass("AdaptativeAttach",true);
#endif
#endif
helper::Creator<InteractionPerformer::InteractionPerformerFactory, FixParticlePerformer<defaulttype::Vec3fTypes> >  FixParticlePerformerVec3fClass("FixParticle",true);
helper::Creator<InteractionPerformer::InteractionPerformerFactory, RemovePrimitivePerformer<defaulttype::Vec3fTypes> >  RemovePrimitivePerformerVec3fClass("RemovePrimitive",true);
helper::Creator<InteractionPerformer::InteractionPerformerFactory, SuturePointPerformer<defaulttype::Vec3fTypes> >  SuturePointPerformerVec3fClass("SuturePoints",true);
#endif
#ifndef SOFA_FLOAT
helper::Creator<InteractionPerformer::InteractionPerformerFactory, AttachBodyPerformer<defaulttype::Vec3dTypes> >  AttachBodyPerformerVec3dClass("AttachBody",true);
#ifdef SOFA_DEV
helper::Creator<InteractionPerformer::InteractionPerformerFactory, AddFramePerformer<defaulttype::Vec3dTypes> >  AddFramePerformerVec3dClass("AddFrame",true);
#ifdef SOFA_HAVE_ARPLUGIN
helper::Creator<InteractionPerformer::InteractionPerformerFactory, AdaptativeAttachPerformer<defaulttype::Vec3dTypes> >  AdaptativeAttachPerformerVec3dClass("AdaptativeAttach",true);
#endif
#endif
helper::Creator<InteractionPerformer::InteractionPerformerFactory, FixParticlePerformer<defaulttype::Vec3dTypes> >  FixParticlePerformerVec3dClass("FixParticle",true);
helper::Creator<InteractionPerformer::InteractionPerformerFactory, RemovePrimitivePerformer<defaulttype::Vec3dTypes> >  RemovePrimitivePerformerVec3dClass("RemovePrimitive",true);
helper::Creator<InteractionPerformer::InteractionPerformerFactory, SuturePointPerformer<defaulttype::Vec3dTypes> >  SuturePointPerformerVec3dClass("SuturePoints",true);
#endif
helper::Creator<InteractionPerformer::InteractionPerformerFactory, InciseAlongPathPerformer>  InciseAlongPathPerformerClass("InciseAlongPath");
helper::Creator<InteractionPerformer::InteractionPerformerFactory, AddRecordedCameraPerformer> AddRecordedCameraPerformerClass("AddRecordedCamera");
helper::Creator<InteractionPerformer::InteractionPerformerFactory, StartNavigationPerformer> StartNavigationPerformerClass("StartNavigation");
#endif

namespace gui
{

//*******************************************************************************************
void Operation::start()
{
    if (!performer)
    {
        performer = createPerformer();
        if (!performer)
        {
            msg_error("MouseOperation") << defaultPerformerType() << " performer cannot be created with the picked model.";
            return;
        }
        else
        {
            pickHandle->getInteraction()->mouseInteractor->addInteractionPerformer(performer);
            configurePerformer(performer);
            performer->start();
        }
    }
}

sofa::component::collision::InteractionPerformer *Operation::createPerformer()
{
    std::string type = defaultPerformerType();
    if (type.empty()) return NULL;
    return component::collision::InteractionPerformer::InteractionPerformerFactory::getInstance()->createObject(type, pickHandle->getInteraction()->mouseInteractor.get());
}

void Operation::configurePerformer(sofa::component::collision::InteractionPerformer* p)
{
    if (mbsetting) p->configure(mbsetting.get());
}

void Operation::end()
{
    if (performer)
    {
        pickHandle->getInteraction()->mouseInteractor->removeInteractionPerformer(performer);
        delete performer; performer=NULL;
    }
}

//*******************************************************************************************
std::string AttachOperation::defaultPerformerType() { return "AttachBody"; }

void AttachOperation::configurePerformer(sofa::component::collision::InteractionPerformer* p)
{
    Operation::configurePerformer(p);
    /*
        //Configuration
        component::collision::AttachBodyPerformerConfiguration *performerConfiguration=dynamic_cast<component::collision::AttachBodyPerformerConfiguration*>(p);
        if (performerConfiguration)
        {
            performerConfiguration->setStiffness(getStiffness());
            performerConfiguration->setArrowSize(getArrowSize());
            performerConfiguration->setShowFactorSize(getShowFactorSize());
        }
    */
}

//*******************************************************************************************
std::string FixOperation::defaultPerformerType() { return "FixParticle"; }

void FixOperation::configurePerformer(sofa::component::collision::InteractionPerformer* performer)
{
    Operation::configurePerformer(performer);
    //Configuration
    component::collision::FixParticlePerformerConfiguration *performerConfiguration=dynamic_cast<component::collision::FixParticlePerformerConfiguration*>(performer);
    performerConfiguration->setStiffness(getStiffness());
}

//*******************************************************************************************
void TopologyOperation::start()
{
    if (getTopologicalOperation() == 0)  // Remove one element
    {
        performer=component::collision::InteractionPerformer::InteractionPerformerFactory::getInstance()->createObject("RemovePrimitive", pickHandle->getInteraction()->mouseInteractor.get());
        pickHandle->getInteraction()->mouseInteractor->addInteractionPerformer(performer);

        performer->start();
    }
    else if (getTopologicalOperation() == 1)
    {
        if (firstClick)
        {
            performer=component::collision::InteractionPerformer::InteractionPerformerFactory::getInstance()->createObject("RemovePrimitive", pickHandle->getInteraction()->mouseInteractor.get());
            pickHandle->getInteraction()->mouseInteractor->addInteractionPerformer(performer);

            component::collision::RemovePrimitivePerformerConfiguration *performerConfiguration=dynamic_cast<component::collision::RemovePrimitivePerformerConfiguration*>(performer);

            performerConfiguration->setTopologicalOperation( getTopologicalOperation() );
            performerConfiguration->setVolumicMesh( getVolumicMesh() );
            performerConfiguration->setScale( getScale() );

            performer->start();
            firstClick = false;
        }
        else
        {
            performer->start();
            firstClick = true;
        }
    }
}

void TopologyOperation::execution()
{
    //       performer->execute();
}

void TopologyOperation::end()
{
    if (getTopologicalOperation() == 0 || (getTopologicalOperation() == 1 && firstClick))
    {
        pickHandle->getInteraction()->mouseInteractor->removeInteractionPerformer(performer);
        delete performer; performer=0;
    }

}

void TopologyOperation::endOperation()
{
    if (performer)
    {
        pickHandle->getInteraction()->mouseInteractor->removeInteractionPerformer(performer);
        delete performer; performer=0;
        firstClick = true;
    }
}



//*******************************************************************************************
void InciseOperation::start()
{
    int currentMethod = getIncisionMethod();

    if (!startPerformer)
    {
        startPerformer=component::collision::InteractionPerformer::InteractionPerformerFactory::getInstance()->createObject("InciseAlongPath", pickHandle->getInteraction()->mouseInteractor.get());
        component::collision::InciseAlongPathPerformerConfiguration *performerConfigurationStart=dynamic_cast<component::collision::InciseAlongPathPerformerConfiguration*>(startPerformer);
        performerConfigurationStart->setIncisionMethod(getIncisionMethod());
        performerConfigurationStart->setSnapingBorderValue(getSnapingBorderValue());
        performerConfigurationStart->setSnapingValue(getSnapingValue());

        pickHandle->getInteraction()->mouseInteractor->addInteractionPerformer(startPerformer);
        startPerformer->setPerformerFreeze();
        startPerformer->start();
    }

    if (currentMethod == 0) // incision clic by clic.
    {
        if (cpt == 0) // First clic => initialisation
        {
            performer=component::collision::InteractionPerformer::InteractionPerformerFactory::getInstance()->createObject("InciseAlongPath", pickHandle->getInteraction()->mouseInteractor.get());

            component::collision::InciseAlongPathPerformerConfiguration *performerConfiguration=dynamic_cast<component::collision::InciseAlongPathPerformerConfiguration*>(performer);
            performerConfiguration->setIncisionMethod(getIncisionMethod());
            performerConfiguration->setSnapingBorderValue(getSnapingBorderValue());
            performerConfiguration->setSnapingValue(getSnapingValue());
            performerConfiguration->setCompleteIncision(getCompleteIncision());
            performerConfiguration->setKeepPoint(getKeepPoint());

            pickHandle->getInteraction()->mouseInteractor->addInteractionPerformer(performer);
            performer->start();
            cpt++;
        }
        else // Second clic(s) only perform start() method
        {
            performer->start();
            startPerformer->start();
        }
    }
    else
    {
        if (cpt != 0)
        {
            pickHandle->getInteraction()->mouseInteractor->removeInteractionPerformer(performer);
            delete performer; performer=0;
        }
        performer=component::collision::InteractionPerformer::InteractionPerformerFactory::getInstance()->createObject("InciseAlongPath", pickHandle->getInteraction()->mouseInteractor.get());

        component::collision::InciseAlongPathPerformerConfiguration *performerConfiguration=dynamic_cast<component::collision::InciseAlongPathPerformerConfiguration*>(performer);
        performerConfiguration->setIncisionMethod(getIncisionMethod());
        performerConfiguration->setSnapingBorderValue(getSnapingBorderValue());
        performerConfiguration->setSnapingValue(getSnapingValue());

        pickHandle->getInteraction()->mouseInteractor->addInteractionPerformer(performer);
        performer->start();
        cpt++;
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
    // On fini la coupure ici!
    if (getCompleteIncision() && startPerformer)
    {
        startPerformer->setPerformerFreeze();

        //Remove startPerformer
        pickHandle->getInteraction()->mouseInteractor->removeInteractionPerformer(startPerformer);
        delete startPerformer; startPerformer=0;
    }

    if (!getKeepPoint())
    {
        cpt = 0; //reinitialization
        pickHandle->getInteraction()->mouseInteractor->removeInteractionPerformer(performer);
        delete performer; performer=0;
    }
}

InciseOperation::~InciseOperation()
{
    if (performer)
    {
        pickHandle->getInteraction()->mouseInteractor->removeInteractionPerformer(performer);
        delete performer; performer=0;
    }
}


//*******************************************************************************************
std::string AddFrameOperation::defaultPerformerType() { return "AddFrame"; }

void AddFrameOperation::configurePerformer(sofa::component::collision::InteractionPerformer* p)
{
    Operation::configurePerformer(p);
}


//*******************************************************************************************
std::string AddRecordedCameraOperation::defaultPerformerType() { return "AddRecordedCamera"; }

void AddRecordedCameraOperation::configurePerformer(sofa::component::collision::InteractionPerformer* p)
{
    Operation::configurePerformer(p);
}


//*******************************************************************************************
std::string StartNavigationOperation::defaultPerformerType() { return "StartNavigation"; }

void StartNavigationOperation::configurePerformer(sofa::component::collision::InteractionPerformer* p)
{
    Operation::configurePerformer(p);
}


//*******************************************************************************************
std::string AddSutureOperation::defaultPerformerType() { return "SuturePoints"; }

void AddSutureOperation::configurePerformer(sofa::component::collision::InteractionPerformer* performer)
{
    Operation::configurePerformer(performer);
    //configuration
    component::collision::SuturePointPerformerConfiguration *performerConfiguration=dynamic_cast<component::collision::SuturePointPerformerConfiguration*>(performer);
    if (performerConfiguration)
    {
        performerConfiguration->setStiffness(getStiffness());
        performerConfiguration->setDamping(getDamping());
    }
}

}
}
