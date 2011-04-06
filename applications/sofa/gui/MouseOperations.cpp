/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/gui/MouseOperations.h>
#include <sofa/gui/PickHandler.h>
#include <sofa/component/collision/InteractionPerformer.h>

#include <sofa/component/collision/ComponentMouseInteraction.h>
#include <sofa/component/collision/AttachBodyPerformer.h>
#include <sofa/component/collision/FixParticlePerformer.h>
#include <sofa/component/collision/RemovePrimitivePerformer.h>
#include <sofa/component/collision/InciseAlongPathPerformer.h>
#include <sofa/component/collision/AddFramePerformer.h>
#include <sofa/component/collision/SuturePointPerformer.h>


namespace sofa
{

using namespace component::collision;

#ifdef WIN32
#ifndef SOFA_DOUBLE
helper::Creator<InteractionPerformer::InteractionPerformerFactory, AttachBodyPerformer<defaulttype::Vec3fTypes> >  AttachBodyPerformerVec3fClass("AttachBody",true);
helper::Creator<InteractionPerformer::InteractionPerformerFactory, AddFramePerformer<defaulttype::Vec3fTypes> >  AddFramePerformerVec3fClass("AddFrame",true);
helper::Creator<InteractionPerformer::InteractionPerformerFactory, FixParticlePerformer<defaulttype::Vec3fTypes> >  FixParticlePerformerVec3fClass("FixParticle",true);
helper::Creator<InteractionPerformer::InteractionPerformerFactory, RemovePrimitivePerformer<defaulttype::Vec3fTypes> >  RemovePrimitivePerformerVec3fClass("RemovePrimitive",true);
helper::Creator<InteractionPerformer::InteractionPerformerFactory, SuturePointPerformer<defaulttype::Vec3fTypes> >  SuturePointPerformerVec3fClass("SuturePoints",true);
#endif
#ifndef SOFA_FLOAT
helper::Creator<InteractionPerformer::InteractionPerformerFactory, AttachBodyPerformer<defaulttype::Vec3dTypes> >  AttachBodyPerformerVec3dClass("AttachBody",true);
helper::Creator<InteractionPerformer::InteractionPerformerFactory, AddFramePerformer<defaulttype::Vec3dTypes> >  AddFramePerformerVec3dClass("AddFrame",true);
helper::Creator<InteractionPerformer::InteractionPerformerFactory, FixParticlePerformer<defaulttype::Vec3dTypes> >  FixParticlePerformerVec3dClass("FixParticle",true);
helper::Creator<InteractionPerformer::InteractionPerformerFactory, RemovePrimitivePerformer<defaulttype::Vec3dTypes> >  RemovePrimitivePerformerVec3dClass("RemovePrimitive",true);
helper::Creator<InteractionPerformer::InteractionPerformerFactory, SuturePointPerformer<defaulttype::Vec3dTypes> >  SuturePointPerformerVec3dClass("SuturePoints",true);
#endif
helper::Creator<InteractionPerformer::InteractionPerformerFactory, InciseAlongPathPerformer>  InciseAlongPathPerformerClass("InciseAlongPath");
#endif

namespace gui
{

//*******************************************************************************************
void AttachOperation::start()
{
    if (!performer)
    {
        //Creation
        performer=component::collision::InteractionPerformer::InteractionPerformerFactory::getInstance()->createObject("AttachBody", pickHandle->getInteraction()->mouseInteractor);
        pickHandle->getInteraction()->mouseInteractor->addInteractionPerformer(performer);
        //Configuration
        component::collision::AttachBodyPerformerConfiguration *performerConfiguration=dynamic_cast<component::collision::AttachBodyPerformerConfiguration*>(performer);
        performerConfiguration->setStiffness(getStiffness());
        performerConfiguration->setArrowSize(getArrowSize());
        performerConfiguration->setShowFactorSize(getShowFactorSize());
        //Start
        performer->start();
    }
}

void AttachOperation::execution()
{
    //do nothing
}

void AttachOperation::end()
{
    pickHandle->getInteraction()->mouseInteractor->removeInteractionPerformer(performer);
    delete performer; performer=0;
}

void AttachOperation::endOperation()
{
    pickHandle->getInteraction()->mouseInteractor->removeInteractionPerformer(performer);
}


//*******************************************************************************************
void FixOperation::start()
{
    //Creation
    performer=component::collision::InteractionPerformer::InteractionPerformerFactory::getInstance()->createObject("FixParticle", pickHandle->getInteraction()->mouseInteractor);
    if (!performer) {std::cerr << "FixParticule performer is not able to run with the model picked" << std::endl; return;}
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
void TopologyOperation::start()
{
    //std::cout <<"TopologyOperation::start()"<< std::endl;

    if (getTopologicalOperation() == 0)  // Remove one element
    {
        performer=component::collision::InteractionPerformer::InteractionPerformerFactory::getInstance()->createObject("RemovePrimitive", pickHandle->getInteraction()->mouseInteractor);
        pickHandle->getInteraction()->mouseInteractor->addInteractionPerformer(performer);

        performer->start();
    }
    else if (getTopologicalOperation() == 1)
    {
        if (firstClick)
        {
            performer=component::collision::InteractionPerformer::InteractionPerformerFactory::getInstance()->createObject("RemovePrimitive", pickHandle->getInteraction()->mouseInteractor);
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
        startPerformer=component::collision::InteractionPerformer::InteractionPerformerFactory::getInstance()->createObject("InciseAlongPath", pickHandle->getInteraction()->mouseInteractor);
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
            performer=component::collision::InteractionPerformer::InteractionPerformerFactory::getInstance()->createObject("InciseAlongPath", pickHandle->getInteraction()->mouseInteractor);

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
        performer=component::collision::InteractionPerformer::InteractionPerformerFactory::getInstance()->createObject("InciseAlongPath", pickHandle->getInteraction()->mouseInteractor);

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



void AddFrameOperation::start()
{
    //Creation
    performer=component::collision::InteractionPerformer::InteractionPerformerFactory::getInstance()->createObject("AddFrame", pickHandle->getInteraction()->mouseInteractor);
    pickHandle->getInteraction()->mouseInteractor->addInteractionPerformer(performer);
    //Start
    performer->start();
}



//*******************************************************************************************
void AddSutureOperation::start()
{
    //Creation
    if (!performer) //first clic
    {
        performer = component::collision::InteractionPerformer::InteractionPerformerFactory::getInstance()->createObject("SuturePoints", pickHandle->getInteraction()->mouseInteractor);
        pickHandle->getInteraction()->mouseInteractor->addInteractionPerformer(performer);

        //configuration
        component::collision::SuturePointPerformerConfiguration *performerConfiguration=dynamic_cast<component::collision::SuturePointPerformerConfiguration*>(performer);
        performerConfiguration->setStiffness(getStiffness());
        performerConfiguration->setDamping(getDamping());
    }

    //Start
    performer->start();
}


void AddSutureOperation::endOperation()
{
    pickHandle->getInteraction()->mouseInteractor->removeInteractionPerformer(performer);
    delete performer; performer=0;
}
}
}
