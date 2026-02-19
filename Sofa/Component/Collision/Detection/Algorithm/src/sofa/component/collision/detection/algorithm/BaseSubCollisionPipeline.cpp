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
#include <sofa/core/collision/Contact.h>

namespace sofa::component::collision::detection::algorithm
{

BaseSubCollisionPipeline::BaseSubCollisionPipeline()
: sofa::core::objectmodel::BaseObject()
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
    this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Loading);

    doInit();
}

/**
 * @brief Queries all registered contact response types from the Contact factory.
 *
 * This static method iterates through all contact types registered in the
 * Contact::Factory and returns their names. These represent the available
 * collision response methods (e.g., "PenalityContactForceField", "FrictionContact").
 *
 * @return A set of strings containing all registered contact response type names.
 */
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
