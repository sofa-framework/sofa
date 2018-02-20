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
#include "MouseOperations.h"
#include <sofa/gui/PickHandler.h>
#include <SofaUserInteraction/ComponentMouseInteraction.h>
#include <PhysicsBasedInteractiveModeler/pim/SculptBodyPerformer.h>

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
    performerConfiguration->setCheckedInflate(isCheckedInflate());
    performerConfiguration->setCheckedDeflate(isCheckedDeflate());
    performerConfiguration->setForce(getForce()/50000);
    SculptBodyPerformer<Vec3Types>* sculptPerformer=dynamic_cast<SculptBodyPerformer<Vec3Types>*>(performer);
    sculptPerformer->start();

    performerConfiguration->setMass(getMass());
    performerConfiguration->setStiffness(getStiffness());
    performerConfiguration->setDamping(getDamping());
}

void SculptOperation::end()
{
    if (performer == NULL) return;

    SculptBodyPerformerConfiguration *performerConfiguration=dynamic_cast<SculptBodyPerformerConfiguration*>(performer);
    performerConfiguration->setForce(0.0);
    performerConfiguration->setCheckedFix(false);
    SculptBodyPerformer<Vec3Types>* sculptPerformer=dynamic_cast<SculptBodyPerformer<Vec3Types>*>(performer);
    sculptPerformer->end();
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
}

} // namespace gui
} // namespace pim
} // namespace plugins
