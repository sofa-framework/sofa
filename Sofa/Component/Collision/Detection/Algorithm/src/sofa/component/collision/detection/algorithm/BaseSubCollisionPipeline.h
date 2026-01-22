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

#include <sofa/core/CollisionModel.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/collision/ContactManager.h>
#include <sofa/core/collision/Intersection.h>

#include <set>
#include <string>

namespace sofa::component::collision::detection::algorithm
{

class SOFA_COMPONENT_COLLISION_DETECTION_ALGORITHM_API BaseSubCollisionPipeline : public sofa::core::objectmodel::BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(BaseSubCollisionPipeline, sofa::core::objectmodel::BaseObject);

protected:
    BaseSubCollisionPipeline();
    
    virtual void doInit() = 0;
    virtual void doBwdInit();
    virtual void doHandleEvent(sofa::core::objectmodel::Event* e) = 0;
    virtual void doDraw(const core::visual::VisualParams* vparams);
    
public:
    virtual void computeCollisionReset() = 0;
    virtual void computeCollisionDetection() = 0;
    virtual void computeCollisionResponse() = 0;
    
    void init() override final;
    void draw(const core::visual::VisualParams* vparams) override final;
    void handleEvent(sofa::core::objectmodel::Event* e) override final;
    
    static std::set< std::string > getResponseList();

    sofa::MultiLink < BaseSubCollisionPipeline, sofa::core::CollisionModel, sofa::BaseLink::FLAG_DUPLICATE > l_collisionModels;
    sofa::SingleLink< BaseSubCollisionPipeline, sofa::core::collision::Intersection, sofa::BaseLink::FLAG_STOREPATH | sofa::BaseLink::FLAG_STRONGLINK > l_intersectionMethod;
    sofa::SingleLink< BaseSubCollisionPipeline, sofa::core::collision::ContactManager, sofa::BaseLink::FLAG_STOREPATH | sofa::BaseLink::FLAG_STRONGLINK > l_contactManager;
};

} // namespace sofa::component::collision::detection::algorithm
