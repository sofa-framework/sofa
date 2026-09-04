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
#include <sofa/core/collision/Pipeline.h>
#include <sofa/core/objectmodel/BaseNode.h>

namespace sofa::core::collision
{
//using namespace core::objectmodel;
//using namespace core::behavior;

Pipeline::Pipeline()
    : intersectionMethod(nullptr),
      broadPhaseDetection(nullptr),
      narrowPhaseDetection(nullptr),
      contactManager(nullptr),
      groupManager(nullptr)
{
}

Pipeline::~Pipeline()
{
}


const BroadPhaseDetection *Pipeline::getBroadPhaseDetection() const
{
    return broadPhaseDetection;
}


const NarrowPhaseDetection *Pipeline::getNarrowPhaseDetection() const
{
    return narrowPhaseDetection;
}

bool Pipeline::insertInNode( objectmodel::BaseNode* node )
{
    node->addCollisionPipeline(this);
    Inherit1::insertInNode(node);
    return true;
}

bool Pipeline::removeInNode( objectmodel::BaseNode* node )
{
    node->removeCollisionPipeline(this);
    Inherit1::removeInNode(node);
    return true;
}

/// Entry point for collision reset, called by the simulation loop. It removes collision response from last step
void Pipeline::computeCollisionReset()
{
    //TODO (SPRINT SED 2025): Component state mechamism
    if(!this->isComponentStateValid())
        return;

    doComputeCollisionReset();
}

/// Entry point for collision detection, called by the simulation loop. Note that this step must not modify the simulation graph
void Pipeline::computeCollisionDetection()
{
    //TODO (SPRINT SED 2025): Component state mechamism
    if(!this->isComponentStateValid())
        return;

    // The collision models parameter is not used by this pipeline
    // since each sub-pipeline manages its own set of models
    static std::vector<CollisionModel*> collisionModels{};

    doComputeCollisionDetection(collisionModels);
}

/// Entry point for collision response, called by the simulation loop. It adds the collision response in the simulation graph
void Pipeline::computeCollisionResponse()
{
    //TODO (SPRINT SED 2025): Component state mechamism
    if(!this->isComponentStateValid())
        return;

    doComputeCollisionResponse();
}

} // namespace sofa

