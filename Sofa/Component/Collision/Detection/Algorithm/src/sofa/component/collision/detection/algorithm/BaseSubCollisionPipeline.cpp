/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include <sofa/component/collision/detection/algorithm/BaseSubCollisionPipeline.h>

#include <sofa/core/visual/VisualParams.h>

namespace sofa::component::collision::detection::algorithm
{

BaseSubCollisionPipeline::BaseSubCollisionPipeline()
: sofa::core::objectmodel::BaseObject()
, l_collisionModels(initLink("collisionModels", "List of collision models to consider in this pipeline"))
, l_intersectionMethod(initLink("intersectionMethod", "Intersection method to use in this pipeline"))
, l_contactManager(initLink("contactManager", "Contact manager to use in this pipeline"))
{
    
}

void BaseSubCollisionPipeline::doBwdInit()
{
    
}

void BaseSubCollisionPipeline::doDraw(const core::visual::VisualParams* vparams)
{
    SOFA_UNUSED(vparams);
    
}

void BaseSubCollisionPipeline::init()
{
    bool validity = true;
    
    this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
    
    //Check given parameters
    if (l_collisionModels.size() == 0)
    {
        msg_warning() << "At least one CollisionModel is required to compute collision detection.";
        validity = false;
    }
    
    if (!l_intersectionMethod)
    {
        msg_warning() << "An Intersection detection component is required to compute collision detection.";
        validity = false;
    }
    
    if (!l_contactManager)
    {
        msg_warning() << "A contact manager component is required to compute collision detection.";
        validity = false;
    }
    
    if (validity)
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
    }
            
    doInit();
}

std::set< std::string > BaseSubCollisionPipeline::getResponseList()
{
    std::set< std::string > listResponse;
    for (const auto& [key, creatorPtr] : *core::collision::Contact::Factory::getInstance())
    {
        listResponse.insert(key);
    }
    return listResponse;
}

void BaseSubCollisionPipeline::draw(const core::visual::VisualParams* vparams)
{
    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    doDraw(vparams);
}

void BaseSubCollisionPipeline::handleEvent(sofa::core::objectmodel::Event* e)
{
    doHandleEvent(e);
}

} // namespace sofa::component::collision::detection::algorithm
