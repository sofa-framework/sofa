/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <sofa/gui/common/MouseOperations.h>
#include <sofa/gui/common/PickHandler.h>

#include <sofa/gui/component/performer/InteractionPerformer.h>
#include <sofa/gui/component/performer/ComponentMouseInteraction.h>
#include <sofa/gui/component/performer/AttachBodyPerformer.h>
#include <sofa/gui/component/performer/FixParticlePerformer.h>
#include <sofa/gui/component/performer/RemovePrimitivePerformer.h>
#include <sofa/gui/component/performer/InciseAlongPathPerformer.h>
#include <sofa/gui/component/performer/AddRecordedCameraPerformer.h>
#include <sofa/gui/component/performer/StartNavigationPerformer.h>
#include <sofa/gui/component/performer/SuturePointPerformer.h>

namespace sofa
{
using namespace sofa::gui::component::performer;

#ifdef WIN32
helper::Creator<InteractionPerformer::InteractionPerformerFactory, AttachBodyPerformer<defaulttype::Vec3Types> >  AttachBodyPerformerVec3Class("AttachBody",true);
helper::Creator<InteractionPerformer::InteractionPerformerFactory, FixParticlePerformer<defaulttype::Vec3Types> >  FixParticlePerformerVec3Class("FixParticle",true);
helper::Creator<InteractionPerformer::InteractionPerformerFactory, RemovePrimitivePerformer<defaulttype::Vec3Types> >  RemovePrimitivePerformerVec3Class("RemovePrimitive",true);
helper::Creator<InteractionPerformer::InteractionPerformerFactory, SuturePointPerformer<defaulttype::Vec3Types> >  SuturePointPerformerVec3Class("SuturePoints",true);
#endif
} // namespace sofa

namespace sofa::gui::common
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

InteractionPerformer *Operation::createPerformer()
{
    const std::string type = defaultPerformerType();
    if (type.empty()) return nullptr;
    return InteractionPerformer::InteractionPerformerFactory::getInstance()->createObject(type, pickHandle->getInteraction()->mouseInteractor.get());
}

void Operation::configurePerformer(InteractionPerformer* p)
{
    if (mbsetting) p->configure(mbsetting.get());
}

void Operation::end()
{
    if (performer)
    {
        pickHandle->getInteraction()->mouseInteractor->removeInteractionPerformer(performer);
        delete performer; performer=nullptr;
    }
}

//*******************************************************************************************
std::string AttachOperation::defaultPerformerType() { return "AttachBody"; }

void AttachOperation::configurePerformer(InteractionPerformer* p)
{
    Operation::configurePerformer(p);
    /*
        //Configuration
        AttachBodyPerformerConfiguration *performerConfiguration=dynamic_cast<AttachBodyPerformerConfiguration*>(p);
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

void FixOperation::configurePerformer(InteractionPerformer* performer)
{
    Operation::configurePerformer(performer);
    //Configuration
    FixParticlePerformerConfiguration *performerConfiguration=dynamic_cast<FixParticlePerformerConfiguration*>(performer);
    performerConfiguration->setStiffness(getStiffness());
}

//*******************************************************************************************
void TopologyOperation::start()
{
    if (getTopologicalOperation() == 0)  // Remove one element
    {
        performer=InteractionPerformer::InteractionPerformerFactory::getInstance()->createObject("RemovePrimitive", pickHandle->getInteraction()->mouseInteractor.get());
        pickHandle->getInteraction()->mouseInteractor->addInteractionPerformer(performer);

        performer->start();
    }
    else if (getTopologicalOperation() == 1)
    {
        if (firstClick)
        {
            performer=InteractionPerformer::InteractionPerformerFactory::getInstance()->createObject("RemovePrimitive", pickHandle->getInteraction()->mouseInteractor.get());
            pickHandle->getInteraction()->mouseInteractor->addInteractionPerformer(performer);

            RemovePrimitivePerformerConfiguration *performerConfiguration=dynamic_cast<RemovePrimitivePerformerConfiguration*>(performer);

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
        delete performer; performer=nullptr;
    }

}

void TopologyOperation::endOperation()
{
    if (performer)
    {
        pickHandle->getInteraction()->mouseInteractor->removeInteractionPerformer(performer);
        delete performer; performer=nullptr;
        firstClick = true;
    }
}



//*******************************************************************************************
void InciseOperation::start()
{
    const int currentMethod = getIncisionMethod();

    if (!startPerformer)
    {
        startPerformer=InteractionPerformer::InteractionPerformerFactory::getInstance()->createObject("InciseAlongPath", pickHandle->getInteraction()->mouseInteractor.get());
        InciseAlongPathPerformerConfiguration *performerConfigurationStart=dynamic_cast<InciseAlongPathPerformerConfiguration*>(startPerformer);
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
            performer=InteractionPerformer::InteractionPerformerFactory::getInstance()->createObject("InciseAlongPath", pickHandle->getInteraction()->mouseInteractor.get());

            InciseAlongPathPerformerConfiguration *performerConfiguration=dynamic_cast<InciseAlongPathPerformerConfiguration*>(performer);
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
            delete performer; performer=nullptr;
        }
        performer=InteractionPerformer::InteractionPerformerFactory::getInstance()->createObject("InciseAlongPath", pickHandle->getInteraction()->mouseInteractor.get());

        InciseAlongPathPerformerConfiguration *performerConfiguration=dynamic_cast<InciseAlongPathPerformerConfiguration*>(performer);
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
        delete startPerformer; startPerformer=nullptr;
    }

    if (!getKeepPoint())
    {
        cpt = 0; //reinitialization
        pickHandle->getInteraction()->mouseInteractor->removeInteractionPerformer(performer);
        delete performer; performer=nullptr;
    }
}

InciseOperation::~InciseOperation()
{
    if (performer)
    {
        pickHandle->getInteraction()->mouseInteractor->removeInteractionPerformer(performer);
        delete performer; performer=nullptr;
    }
}


//*******************************************************************************************
std::string AddFrameOperation::defaultPerformerType() { return "AddFrame"; }

void AddFrameOperation::configurePerformer(InteractionPerformer* p)
{
    Operation::configurePerformer(p);
}


//*******************************************************************************************
std::string AddRecordedCameraOperation::defaultPerformerType() { return "AddRecordedCamera"; }

void AddRecordedCameraOperation::configurePerformer(InteractionPerformer* p)
{
    Operation::configurePerformer(p);
}


//*******************************************************************************************
std::string StartNavigationOperation::defaultPerformerType() { return "StartNavigation"; }

void StartNavigationOperation::configurePerformer(InteractionPerformer* p)
{
    Operation::configurePerformer(p);
}


//*******************************************************************************************
std::string AddSutureOperation::defaultPerformerType() { return "SuturePoints"; }

void AddSutureOperation::configurePerformer(InteractionPerformer* performer)
{
    Operation::configurePerformer(performer);
    //configuration
    SuturePointPerformerConfiguration *performerConfiguration=dynamic_cast<SuturePointPerformerConfiguration*>(performer);
    if (performerConfiguration)
    {
        performerConfiguration->setStiffness(getStiffness());
        performerConfiguration->setDamping(getDamping());
    }
}

} // namespace sofa::gui::common
