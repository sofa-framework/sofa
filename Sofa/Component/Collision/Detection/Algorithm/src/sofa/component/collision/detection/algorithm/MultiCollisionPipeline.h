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

namespace sofa::simulation
{
class TaskScheduler;
}

namespace sofa::component::collision::detection::algorithm
{

class AbstractSubCollisionPipeline;

class SOFA_COMPONENT_COLLISION_DETECTION_ALGORITHM_API MultiCollisionPipeline : public sofa::core::collision::Pipeline
{
public:
    SOFA_CLASS(MultiCollisionPipeline, sofa::core::collision::Pipeline);

    sofa::Data<int>  d_depth;
protected:
    MultiCollisionPipeline();
public:
    void init() override;
    void bwdInit() override;

    /// get the set of response available with the current collision pipeline
    std::set< std::string > getResponseList() const override;
protected:
    // -- Pipeline interface
    /// Remove collision response from last step
    void doCollisionReset() override;
    /// Detect new collisions. Note that this step must not modify the simulation graph
    void doCollisionDetection(const sofa::type::vector<sofa::core::CollisionModel*>& collisionModels) override;
    /// Add collision response in the simulation graph
    void doCollisionResponse() override;

    void reset() override;
    
    void draw(const core::visual::VisualParams* vparams) override;

    /// Remove collision response from last step
    virtual void computeCollisionReset() override;
    /// Detect new collisions. Note that this step must not modify the simulation graph
    virtual void computeCollisionDetection() override;
    /// Add collision response in the simulation graph
    virtual void computeCollisionResponse() override;
    
    sofa::simulation::TaskScheduler* m_taskScheduler{nullptr};
    
    std::vector<AbstractSubCollisionPipeline*> m_subCollisionPipelines;

public:
    sofa::Data<bool> d_parallelDetection;
    sofa::Data<bool> d_parallelResponse;
    sofa::MultiLink < MultiCollisionPipeline, AbstractSubCollisionPipeline, sofa::BaseLink::FLAG_DUPLICATE > l_subCollisionPipelines;
};

} // namespace sofa::component::collision::detection::algorithm
