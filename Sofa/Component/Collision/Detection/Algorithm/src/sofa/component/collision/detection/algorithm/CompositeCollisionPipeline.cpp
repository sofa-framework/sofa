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
#include <sofa/component/collision/detection/algorithm/CompositeCollisionPipeline.h>

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

void registerCompositeCollisionPipeline(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Multiple collision pipelines in one.")
        .add< CompositeCollisionPipeline >());
}

CompositeCollisionPipeline::CompositeCollisionPipeline()
    : d_parallelDetection(initData(&d_parallelDetection, false, "parallelDetection", "Parallelize collision detection."))
    , l_subCollisionPipelines(initLink("subCollisionPipelines", "List of sub collision pipelines to handle."))
{
}

/**
 * @brief Initializes the composite pipeline and validates its configuration.
 *
 * This method performs several validation and setup steps:
 * 1. Validates that at least one sub-pipeline is linked
 * 2. Initializes the task scheduler if parallel detection is enabled
 * 3. Validates all linked sub-pipelines are valid (non-null)
 * 4. Checks that all collision models in the scene are covered by at least one sub-pipeline
 *    (issues warnings for any uncovered models to help users identify configuration issues)
 */
void CompositeCollisionPipeline::init()
{
    Inherit1::init();

    this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);

    // Validate that at least one sub-pipeline is defined
    if(l_subCollisionPipelines.size() == 0)
    {
        msg_warning() << "No SubCollisionPipeline defined in CompositeCollisionPipeline. Nothing will be done." ;

        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    // Initialize task scheduler for parallel execution if enabled
    if(d_parallelDetection.getValue())
    {
        this->initTaskScheduler();
    }

    // Collect all collision models from the scene to verify coverage
    simulation::Node* root = dynamic_cast<simulation::Node*>(getContext());
    std::vector<CollisionModel*> sceneCollisionModels;
    root->getTreeObjects<CollisionModel>(&sceneCollisionModels);

    // Collect all collision models handled by sub-pipelines
    std::set<CollisionModel*> pipelineCollisionModels;
    for(auto* subPipeline : l_subCollisionPipelines)
    {
        if(!subPipeline)
        {
            msg_error() << "One of the subCollisionPipeline is incorrect (nullptr or invalid) ";
            this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
            return;
        }

        for (auto* cm : subPipeline->getCollisionModels())
        {
            pipelineCollisionModels.insert(cm);
        }
    }

    // Warn about collision models not covered by any sub-pipeline
    // This helps users identify configuration issues where some models won't participate in collision
    for (const auto& cm : sceneCollisionModels)
    {
        if (pipelineCollisionModels.find(cm) == pipelineCollisionModels.end())
        {
            msg_warning() << "CollisionModel " << cm->getPathName() << " is not handled by any SubCollisionPipeline.";
        }
    }

}

void CompositeCollisionPipeline::reset()
{

}

/// Delegates collision reset to all sub-pipelines sequentially.
void CompositeCollisionPipeline::doCollisionReset()
{
    msg_info() << "CompositeCollisionPipeline::doCollisionReset" ;

    for(const auto& subPipeline : l_subCollisionPipelines)
    {
        subPipeline->computeCollisionReset();
    }
}

/**
 * @brief Executes collision detection across all sub-pipelines.
 *
 * If parallel detection is enabled and a task scheduler is available, the detection
 * phase of each sub-pipeline runs concurrently. This can significantly improve
 * performance when there are multiple independent collision groups.
 *
 * @param collisionModels Ignored - each sub-pipeline uses its own linked collision models.
 */
void CompositeCollisionPipeline::doCollisionDetection(const type::vector<core::CollisionModel*>& collisionModels)
{
    SOFA_UNUSED(collisionModels);

    SCOPED_TIMER_VARNAME(docollisiontimer, "doCollisionDetection");

    if(m_taskScheduler)
    {
        // Parallel execution: distribute sub-pipeline detection across available threads
        auto computeCollisionDetection = [&](const auto& range)
        {
            for (auto it = range.start; it != range.end; ++it)
            {
                (*it)->computeCollisionDetection();
            }
        };

        sofa::simulation::forEachRange(sofa::simulation::ForEachExecutionPolicy::PARALLEL, *m_taskScheduler, l_subCollisionPipelines.begin(), l_subCollisionPipelines.end(), computeCollisionDetection);
    }
    else
    {
        // Sequential execution: process each sub-pipeline one after another
        for (const auto& subPipeline : l_subCollisionPipelines)
        {
            subPipeline->computeCollisionDetection();
        }
    }
}

/// Delegates collision response creation to all sub-pipelines sequentially.
void CompositeCollisionPipeline::doCollisionResponse()
{
    for (const auto& subPipeline : l_subCollisionPipelines)
    {
        subPipeline->computeCollisionResponse();
    }
}

/// Returns the list of available contact response types from the contact factory.
std::set< std::string > CompositeCollisionPipeline::getResponseList() const
{
    return BaseSubCollisionPipeline::getResponseList();
}

/// Entry point for collision reset phase, called by the simulation loop.
void CompositeCollisionPipeline::computeCollisionReset()
{
    if(!this->isComponentStateValid())
        return;

    doCollisionReset();
}

/// Entry point for collision detection phase, called by the simulation loop.
void CompositeCollisionPipeline::computeCollisionDetection()
{
    if(!this->isComponentStateValid())
        return;

    // The collision models parameter is not used by this pipeline
    // since each sub-pipeline manages its own set of models
    static std::vector<CollisionModel*> collisionModels{};

    doCollisionDetection(collisionModels);
}

/// Entry point for collision response phase, called by the simulation loop.
void CompositeCollisionPipeline::computeCollisionResponse()
{
    if(!this->isComponentStateValid())
        return;

    doCollisionResponse();
}

} // namespace sofa::component::collision::detection::algorithm
