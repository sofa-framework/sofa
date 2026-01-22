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
#include <sofa/component/collision/detection/algorithm/MultiCollisionPipeline.h>

#include <sofa/component/collision/detection/algorithm/BaseSubCollisionPipeline.h>

#include <sofa/core/CollisionModel.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/collision/BroadPhaseDetection.h>
#include <sofa/core/collision/NarrowPhaseDetection.h>
#include <sofa/core/collision/ContactManager.h>

#include <sofa/simulation/Node.h>
#include <sofa/simulation/MainTaskSchedulerFactory.h>
#include <sofa/simulation/ParallelForEach.h>

#include <sofa/helper/ScopedAdvancedTimer.h>
using sofa::helper::ScopedAdvancedTimer ;

#include <sofa/helper/AdvancedTimer.h>


namespace sofa::component::collision::detection::algorithm
{

using namespace sofa;
using namespace sofa::core;
using namespace sofa::core::collision;

void registerMultiCollisionPipeline(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Multiple collision pipelines in one.")
        .add< MultiCollisionPipeline >());
}

MultiCollisionPipeline::MultiCollisionPipeline()
    : d_parallelDetection(initData(&d_parallelDetection, false, "parallelDetection", "Parallelize collision detection."))
    , l_subCollisionPipelines(initLink("subCollisionPipelines", "List of sub collision pipelines to handle."))
{
}

void MultiCollisionPipeline::init()
{
    Inherit1::init();

    this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);

    if(l_subCollisionPipelines.size() == 0)
    {
        msg_warning() << "No SubCollisionPipeline defined in MultiCollisionPipeline. Nothing will be done." ;

        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }
    
    if(d_parallelDetection.getValue())
    {
        this->initTaskScheduler();
    }

    // UX: warn if there is any CollisionModel not handled by any SubCollisionPipeline
    simulation::Node* root = dynamic_cast<simulation::Node*>(getContext()->getRootContext());
    std::vector<CollisionModel*> sceneCollisionModels;
    root->getTreeObjects<CollisionModel>(&sceneCollisionModels);

    std::set<CollisionModel*> pipelineCollisionModels;
    for(auto* subPipeline : l_subCollisionPipelines)
    {
        if(!subPipeline)
        {
            msg_error() << "One of the subCollisionPipeline is incorrect (nullptr or invalid) ";
            this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
            return;
        }
        
        for (auto cm : subPipeline->l_collisionModels)
        {
            pipelineCollisionModels.insert(cm);
        }
    }

    for (const auto& cm : sceneCollisionModels)
    {
        if (pipelineCollisionModels.find(cm) == pipelineCollisionModels.end())
        {
            msg_warning() << "CollisionModel " << cm->getName() << " is not handled by any SubCollisionPipeline.";
        }
    }
    
}

void MultiCollisionPipeline::bwdInit()
{
    for(const auto& subPipeline : l_subCollisionPipelines)
    {
        subPipeline->bwdInit();
    }
}

void MultiCollisionPipeline::reset()
{

}

void MultiCollisionPipeline::doCollisionReset()
{
    msg_info() << "MultiCollisionPipeline::doCollisionReset" ;

    for(const auto& subPipeline : l_subCollisionPipelines)
    {
        subPipeline->computeCollisionReset();
    }
}

void MultiCollisionPipeline::doCollisionDetection(const type::vector<core::CollisionModel*>& collisionModels)
{
    SOFA_UNUSED(collisionModels);

    SCOPED_TIMER_VARNAME(docollisiontimer, "doCollisionDetection");

    if(m_taskScheduler)
    {
        auto computeCollisionDetection = [&](const auto& range)
        {
            for (auto it = range.start; it != range.end; ++it)
            {
                (*it)->computeCollisionDetection();
            }
        };
        
        sofa::simulation::forEachRange(sofa::simulation::ForEachExecutionPolicy::PARALLEL, *m_taskScheduler, m_subCollisionPipelines.begin(), m_subCollisionPipelines.end(), computeCollisionDetection);
    }
    else
    {
        for (const auto& subPipeline : m_subCollisionPipelines)
        {
            subPipeline->computeCollisionDetection();
        }
    }
}

void MultiCollisionPipeline::doCollisionResponse()
{
    for (const auto& subPipeline : m_subCollisionPipelines)
    {
        subPipeline->computeCollisionResponse();
    }
}

std::set< std::string > MultiCollisionPipeline::getResponseList() const
{
    return BaseSubCollisionPipeline::getResponseList();
}

void MultiCollisionPipeline::computeCollisionReset()
{
    if(!this->isComponentStateValid())
        return;
    
    doCollisionReset();
}

void MultiCollisionPipeline::computeCollisionDetection()
{
    if(!this->isComponentStateValid())
        return;
    
    //useless
    std::vector<CollisionModel*> collisionModels;
    
    doCollisionDetection(collisionModels);
}

void MultiCollisionPipeline::computeCollisionResponse()
{
    if(!this->isComponentStateValid())
        return;
    
    doCollisionResponse();
}


void MultiCollisionPipeline::draw(const core::visual::VisualParams* vparams)
{
    for (const auto& subPipeline : m_subCollisionPipelines)
    {
        subPipeline->draw(vparams);
    }
}

} // namespace sofa::component::collision::detection::algorithm
