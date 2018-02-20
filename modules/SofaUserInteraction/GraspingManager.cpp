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
#include "GraspingManager.h"
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/collision/DetectionOutput.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/helper/gl/template.h>

namespace sofa
{

namespace component
{

namespace collision
{

SOFA_DECL_CLASS(GraspingManager)

int GraspingManagerClass = core::RegisterObject("Manager handling Grasping operations between a SphereModel and a TriangleSetModel relying on a TetrahedronSetTopology")
        .add< GraspingManager >()
        ;


GraspingManager::GraspingManager()
    : active( initData(&active, false, "active", "Activate this object.\nNote that this can be dynamically controlled by using a key") )
    , keyEvent( initData(&keyEvent, '1', "key", "key to press to activate this object until the key is released") )
    , keySwitchEvent( initData(&keySwitchEvent, '4', "keySwitch", "key to activate this object until the key is pressed again") )
    , openAngle( initData(&openAngle, 1.0, "openAngle", "angle values to set when tool is opened"))
    , closedAngle( initData(&closedAngle, 0.0, "closedAngle", "angle values to set when tool is closed"))
    , mstateTool(NULL)
    , contactManager(NULL)
    , wasActive(false)
{
    this->f_listening.setValue(true);
}

GraspingManager::~GraspingManager()
{
}

void GraspingManager::init()
{
    std::vector<ToolModel*> models;
    this->getContext()->get<ToolModel>(&models, core::objectmodel::BaseContext::SearchDown);
    for (unsigned int i=0; i<models.size(); i++)
    {
        if (models[i]->getContactResponse() == std::string("stick") || models[i]->getContactResponse() == std::string("StickContactConstraint") || models[i]->getName() == std::string("GraspingToolModel"))
        {
            modelTools.insert(models[i]);
        }
    }
    for (std::set<ToolModel*>::iterator it=modelTools.begin(), itend=modelTools.end(); it != itend; ++it)
        (*it)->setActive(false);
    sout << "GraspingManager: "<<modelTools.size()<<"/"<<models.size()<<" collision models selected."<<sendl;
    mstateTool = getContext()->get<ToolDOFs>(core::objectmodel::BaseContext::SearchDown);
    if (mstateTool) sout << "GraspingManager: tool DOFs found"<<sendl;
    contactManager = getContext()->get<core::collision::ContactManager>();
    if (contactManager) sout << "GraspingManager: ContactManager found"<<sendl;
    sout << "GraspingManager: init OK." << sendl;
}

void GraspingManager::reset()
{
}

void GraspingManager::doGrasp()
{
    bool newActive = active.getValue();
    if (newActive && ! wasActive)
    {
        sout << "GraspingManager activated" << sendl;
        // activate CMs for one iteration
        for (std::set<ToolModel*>::iterator it=modelTools.begin(), itend=modelTools.end(); it != itend; ++it)
            (*it)->setActive(true);
    }
    else
    {
        // deactivate CMs
        for (std::set<ToolModel*>::iterator it=modelTools.begin(), itend=modelTools.end(); it != itend; ++it)
            (*it)->setActive(false);
    }

    if (!newActive && wasActive)
    {
        sout << "GraspingManager released" << sendl;
        // clear existing contacts
        if (contactManager)
        {
            const core::collision::ContactManager::ContactVector& cv = contactManager->getContacts();
            for (core::collision::ContactManager::ContactVector::const_iterator it = cv.begin(), itend = cv.end(); it != itend; ++it)
            {
                core::collision::Contact* c = it->get();
                if (modelTools.count(c->getCollisionModels().first) || modelTools.count(c->getCollisionModels().second))
                    c->setKeepAlive(false);
            }
        }
    }

    if (mstateTool)
    {
        SReal value = (newActive ? closedAngle.getValue() : openAngle.getValue());
        sout << value << sendl;
        helper::WriteAccessor<Data<ToolDOFs::VecCoord> > xData = *mstateTool->write(core::VecCoordId::position());
        ToolDOFs::VecCoord& x = xData.wref();
        if (x.size() >= 1)
            x[0] = value;
        if (x.size() >= 2)
            x[1] = -value;
    }

    wasActive = newActive;
}

void GraspingManager::handleEvent(sofa::core::objectmodel::Event* event)
{
    if (sofa::core::objectmodel::KeypressedEvent::checkEventType(event))
    {
        sofa::core::objectmodel::KeypressedEvent* ev = static_cast<sofa::core::objectmodel::KeypressedEvent*>(event);
        if (ev->getKey() == keyEvent.getValue())
        {
            active.setValue(true);
        }
        else if (ev->getKey() == keySwitchEvent.getValue())
        {
            active.setValue(!active.getValue());
        }
    }
    if (sofa::core::objectmodel::KeyreleasedEvent::checkEventType(event))
    {
        sofa::core::objectmodel::KeyreleasedEvent* ev = static_cast<sofa::core::objectmodel::KeyreleasedEvent*>(event);
        if (ev->getKey() == keyEvent.getValue())
        {
            active.setValue(false);
        }
    }
    else if (/* simulation::AnimateEndEvent* ev = */simulation::AnimateEndEvent::checkEventType(event))
    {
//        if (active.getValue())
        doGrasp();
    }
}

} // namespace collision

} // namespace component

} // namespace sofa
