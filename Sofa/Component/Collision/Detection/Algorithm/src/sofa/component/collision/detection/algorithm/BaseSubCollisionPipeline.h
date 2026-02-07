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

#include <sofa/core/objectmodel/BaseObject.h>

#include <set>
#include <string>

namespace sofa::core
{
class CollisionModel;
}

namespace sofa::component::collision::detection::algorithm
{

/**
 * @brief Abstract base class defining the interface for sub-collision pipelines.
 *
 * This base class is designed to be used with CompositeCollisionPipeline, which
 * aggregates multiple sub-pipelines and can execute them in parallel.
 *
 * @see SubCollisionPipeline for a concrete implementation
 * @see CompositeCollisionPipeline for the aggregator that manages sub-pipelines
 */
class SOFA_COMPONENT_COLLISION_DETECTION_ALGORITHM_API BaseSubCollisionPipeline : public sofa::core::objectmodel::BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(BaseSubCollisionPipeline, sofa::core::objectmodel::BaseObject);

protected:
    BaseSubCollisionPipeline();

    /// @brief Called during initialization. Derived classes must implement validation and setup logic.
    virtual void doInit() = 0;

    /// @brief Called after all objects are initialized. Default implementation is empty.
    virtual void doBwdInit();

    /// @brief Called to handle simulation events. Derived classes must implement event processing.
    virtual void doHandleEvent(sofa::core::objectmodel::Event* e) = 0;

    /// @brief Called during rendering. Default implementation is empty.
    virtual void doDraw(const core::visual::VisualParams* vparams);

public:
    ///@{
    /// @name Collision Pipeline Interface
    /// These methods define the three-phase collision workflow that derived classes must implement.

    /// @brief Clears collision state from the previous time step (contacts, responses).
    virtual void computeCollisionReset() = 0;

    /// @brief Performs collision detection (bounding tree, broad phase, narrow phase).
    virtual void computeCollisionDetection() = 0;

    /// @brief Creates collision responses based on detected contacts.
    virtual void computeCollisionResponse() = 0;

    ///@}

    /// @brief Returns the list of collision models handled by this sub-pipeline.
    virtual std::vector<sofa::core::CollisionModel*> getCollisionModels() = 0;

    /// @brief Initializes the component. Marked final to enforce Template Method pattern.
    void init() override final;
    
    /// @brief Initialization of the component during the bottom-up traversal. Marked final to enforce Template Method pattern.
    void bwdInit() override final;

    /// @brief Renders debug visualization. Marked final to enforce Template Method pattern.
    void draw(const core::visual::VisualParams* vparams) override final;

    /// @brief Processes simulation events. Marked final to enforce Template Method pattern.
    void handleEvent(sofa::core::objectmodel::Event* e) override final;

    /// @brief Returns all available contact response types registered in the Contact factory.
    static std::set< std::string > getResponseList();
};

} // namespace sofa::component::collision::detection::algorithm
