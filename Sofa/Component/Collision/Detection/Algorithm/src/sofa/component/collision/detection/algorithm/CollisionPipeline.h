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

#include <sofa/simulation/PipelineImpl.h>

namespace sofa::component::collision::detection::algorithm
{

class SOFA_COMPONENT_COLLISION_DETECTION_ALGORITHM_API CollisionPipeline : public sofa::simulation::PipelineImpl
{
public:
    SOFA_CLASS(CollisionPipeline,sofa::simulation::PipelineImpl);

    Data<bool> d_doPrintInfoMessage;
    Data<bool> d_doDebugDraw;
    Data<int>  d_depth;
protected:
    CollisionPipeline();
public:
    void init() override;

    /// get the set of response available with the current collision pipeline
    std::set< std::string > getResponseList() const override;
protected:
    // -- Pipeline interface
    /// Remove collision response from last step
    void doCollisionReset() override;
    /// Detect new collisions. Note that this step must not modify the simulation graph
    void doCollisionDetection(const sofa::type::vector<core::CollisionModel*>& collisionModels) override;
    /// Add collision response in the simulation graph
    void doCollisionResponse() override;

    virtual void checkDataValues() ;

public:
    static const int defaultDepthValue;
};

} // namespace sofa::component::collision::detection::algorithm
