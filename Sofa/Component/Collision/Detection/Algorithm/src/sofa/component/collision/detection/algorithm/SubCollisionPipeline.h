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

#include <sofa/component/collision/detection/algorithm/BaseSubCollisionPipeline.h>

#include <sofa/core/CollisionModel.h>
#include <sofa/core/collision/ContactManager.h>
#include <sofa/core/collision/Intersection.h>
#include <sofa/core/collision/BroadPhaseDetection.h>
#include <sofa/core/collision/NarrowPhaseDetection.h>

#include <set>
#include <string>

namespace sofa::component::collision::detection::algorithm
{

class SOFA_COMPONENT_COLLISION_DETECTION_ALGORITHM_API SubCollisionPipeline : public BaseSubCollisionPipeline
{
public:
    using Inherited = BaseSubCollisionPipeline;
    SOFA_CLASS(SubCollisionPipeline, Inherited);
protected:
    SubCollisionPipeline();
public:
    virtual ~SubCollisionPipeline() override = default;
    void doInit() override;
    void doHandleEvent(sofa::core::objectmodel::Event*) override {}

    void computeCollisionReset() override;
    void computeCollisionDetection() override;
    void computeCollisionResponse() override;
    
    std::vector<sofa::core::CollisionModel*> getCollisionModels() override;
    
    sofa::Data<unsigned int>  d_depth;
    sofa::MultiLink < SubCollisionPipeline, sofa::core::CollisionModel, sofa::BaseLink::FLAG_DUPLICATE > l_collisionModels;
    sofa::SingleLink< SubCollisionPipeline, sofa::core::collision::Intersection, sofa::BaseLink::FLAG_STOREPATH | sofa::BaseLink::FLAG_STRONGLINK > l_intersectionMethod;
    sofa::SingleLink< SubCollisionPipeline, sofa::core::collision::ContactManager, sofa::BaseLink::FLAG_STOREPATH | sofa::BaseLink::FLAG_STRONGLINK > l_contactManager;
    sofa::SingleLink< SubCollisionPipeline, sofa::core::collision::BroadPhaseDetection, sofa::BaseLink::FLAG_STOREPATH | sofa::BaseLink::FLAG_STRONGLINK > l_broadPhaseDetection;
    sofa::SingleLink< SubCollisionPipeline, sofa::core::collision::NarrowPhaseDetection, sofa::BaseLink::FLAG_STOREPATH | sofa::BaseLink::FLAG_STRONGLINK > l_narrowPhaseDetection;

    static inline constexpr unsigned int s_defaultDepthValue = 6;
};

} // namespace sofa::component::collision::detection::algorithm
