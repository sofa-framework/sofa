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
#include <sofa/component/collision/detection/algorithm/config.h>

#include <sofa/core/collision/Pipeline.h>

#include <sofa/simulation/task/TaskSchedulerUser.h>

namespace sofa::component::collision::detection::algorithm
{

class BaseSubCollisionPipeline;

/**
 * @brief A collision pipeline that aggregates multiple sub-pipelines using the Composite pattern.
 *
 * CompositeCollisionPipeline enables partitioning collision detection into independent groups,
 * where each group is handled by its own SubCollisionPipeline. This architecture provides
 * several benefits:
 *
 * 1. Modularity: Different collision model groups can use different detection algorithms,
 *    intersection methods, or contact managers
 *
 * 2. Parallelization: When enabled via d_parallelDetection, the collision detection phase
 *    of each sub-pipeline runs concurrently, potentially improving performance on multi-core systems
 *
 * 3. Isolation: Collision models in different sub-pipelines won't generate contacts with each other,
 *    allowing intentional separation of non-interacting object groups
 *
 * @see SubCollisionPipeline
 * @see BaseSubCollisionPipeline
 */
class SOFA_COMPONENT_COLLISION_DETECTION_ALGORITHM_API CompositeCollisionPipeline : public sofa::core::collision::Pipeline, public sofa::simulation::TaskSchedulerUser
{
public:
    SOFA_CLASS2(CompositeCollisionPipeline, sofa::core::collision::Pipeline, sofa::simulation::TaskSchedulerUser);

    sofa::Data<int>  d_depth;
protected:
    CompositeCollisionPipeline();
public:
    /// @brief Initializes the pipeline and validates sub-pipeline configuration.
    void init() override;

    /// @brief Returns the set of available collision response types.
    std::set< std::string > getResponseList() const override;
protected:
    // -- Pipeline interface

    /// @brief Delegates reset to all sub-pipelines to clear previous contacts.
    void doCollisionReset() override;

    /// @brief Delegates collision detection to all sub-pipelines (optionally in parallel).
    /// @note The collisionModels parameter is ignored; each sub-pipeline uses its own models.
    void doCollisionDetection(const sofa::type::vector<sofa::core::CollisionModel*>& collisionModels) override;

    /// @brief Delegates response creation to all sub-pipelines.
    void doCollisionResponse() override;

    void reset() override;

    /// @brief Entry point for collision reset, called by the simulation loop.
    virtual void computeCollisionReset() override final;

    /// @brief Entry point for collision detection, called by the simulation loop.
    virtual void computeCollisionDetection() override final;

    /// @brief Entry point for collision response, called by the simulation loop.
    virtual void computeCollisionResponse() override final;

public:
    /// When true, collision detection across sub-pipelines runs in parallel using the task scheduler.
    sofa::Data<bool> d_parallelDetection;

    /// List of sub-pipelines to aggregate. Each handles an independent set of collision models.
    sofa::MultiLink < CompositeCollisionPipeline, BaseSubCollisionPipeline, sofa::BaseLink::FLAG_DUPLICATE > l_subCollisionPipelines;


    friend class CollisionPipeline; // to be able to call do*()
};

} // namespace sofa::component::collision::detection::algorithm
