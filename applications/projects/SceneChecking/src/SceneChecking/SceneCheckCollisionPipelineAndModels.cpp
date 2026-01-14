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
#include "SceneCheckCollisionPipelineAndModels.h"

#include <sofa/simulation/Node.h>
#include <sofa/core/CollisionModel.h>
#include <sofa/core/collision/Pipeline.h>
#include <sofa/simulation/SceneCheckMainRegistry.h>

namespace sofa::_scenechecking_
{

const bool SceneCheckCollisionPipelineAndModelsRegistered = sofa::simulation::SceneCheckMainRegistry::addToRegistry(SceneCheckCollisionPipelineAndModels::newSPtr());

using sofa::simulation::Node;

const std::string SceneCheckCollisionPipelineAndModels::getName()
{
    return "SceneCheckCollisionPipelineAndModels";
}

const std::string SceneCheckCollisionPipelineAndModels::getDesc()
{
    return "Ensure the consistency of the existence of a collision pipeline, and collision models in the scene.";
}

void SceneCheckCollisionPipelineAndModels::doInit(Node* node)
{
    SOFA_UNUSED(node);
}

void SceneCheckCollisionPipelineAndModels::doCheckOn(Node* node)
{
    const sofa::core::objectmodel::BaseContext* root = node->getContext()->getRootContext();
    
    if(!root)
    {
        return;
    }
    
    sofa::core::collision::Pipeline::SPtr anyPipeline{};
    root->get(anyPipeline, sofa::core::objectmodel::BaseContext::SearchDirection::SearchDown);
    sofa::core::CollisionModel::SPtr anyCollisionModel{};
    root->get(anyCollisionModel, sofa::core::objectmodel::BaseContext::SearchDirection::SearchDown);
    
    if(anyPipeline)
    {
        if(anyCollisionModel)
        {
            // there is a collision pipeline and (at least one) collision model(s), carry on.
        }
        else
        {
            // there is a collision pipeline but no collision model.
            // Either the collision pipeline is superfluous;
            // or the collision model(s) has been forgotten.
            m_message = "There is no collision model in this scene, but there is a collision pipeline. Either add a collision model or remove the collision pipeline.";
        }
    }
    else
    {
        if(anyCollisionModel)
        {
            // At least one collision model has been detected but without any pipeline.
            // Either the collision pipeline has been forgotten;
            // or the collision model(s) is useless.
            m_message = "At least one collision model has been found, but there is no collision pipeline. You may add a collision pipeline (or remove the collision model if no collision detection is expected).";
        }
        else
        {
            // there is no collision pipeline and no collision model, the scene certainly does not involve any collision feature.
        }
    }
}

void SceneCheckCollisionPipelineAndModels::doPrintSummary()
{
    if(!m_message.empty())
    {
        msg_warning(this->getName()) << m_message;
    }
}


} // namespace sofa::_scenechecking_
