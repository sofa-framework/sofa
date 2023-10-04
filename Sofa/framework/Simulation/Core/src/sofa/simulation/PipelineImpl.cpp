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
#include <sofa/simulation/PipelineImpl.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/collision/BroadPhaseDetection.h>
#include <sofa/core/collision/NarrowPhaseDetection.h>
#include <sofa/core/collision/CollisionGroupManager.h>
#include <sofa/core/collision/ContactManager.h>

#include <sofa/simulation/Node.h>

#include <sofa/helper/AdvancedTimer.h>
#include <sofa/helper/ScopedAdvancedTimer.h>


namespace sofa
{

namespace simulation
{


using namespace sofa::core;
using namespace sofa::core::objectmodel;
using namespace sofa::core::behavior;
using namespace sofa::core::collision;

PipelineImpl::PipelineImpl()
{
}

PipelineImpl::~PipelineImpl()
{
}

void PipelineImpl::init()
{
    simulation::Node* root = dynamic_cast<simulation::Node*>(getContext());
    if(root == nullptr) return;

    intersectionMethods.clear();
    root->getTreeObjects<Intersection>(&intersectionMethods);
    intersectionMethod = (intersectionMethods.empty() ? nullptr : intersectionMethods[0]);

    broadPhaseDetections.clear();
    root->getTreeObjects<BroadPhaseDetection>(&broadPhaseDetections);
    broadPhaseDetection = (broadPhaseDetections.empty() ? nullptr : broadPhaseDetections[0]);

    narrowPhaseDetections.clear();
    root->getTreeObjects<NarrowPhaseDetection>(&narrowPhaseDetections);
    narrowPhaseDetection = (narrowPhaseDetections.empty() ? nullptr : narrowPhaseDetections[0]);

    contactManagers.clear();
    root->getTreeObjects<ContactManager>(&contactManagers);
    contactManager = (contactManagers.empty() ? nullptr : contactManagers[0]);

    groupManagers.clear();
    root->getTreeObjects<CollisionGroupManager>(&groupManagers);
    groupManager = (groupManagers.empty() ? nullptr : groupManagers[0]);

    if (intersectionMethod==nullptr)
    {
        msg_warning(this) <<"no intersection component defined. Switching to the DiscreteIntersection component. " << msgendl
                            "To remove this warning, you can add an intersection component to your scene. " << msgendl
                            "More details on the collision pipeline can be found at "
                            "[sofadoc::Collision](https://www.sofa-framework.org/community/doc/simulation-principles/multi-model-representation/collision/).";
        sofa::core::objectmodel::BaseObjectDescription discreteIntersectionDesc("Default Intersection","DiscreteIntersection");
        const sofa::core::objectmodel::BaseObject::SPtr obj = sofa::core::ObjectFactory::CreateObject(getContext(), &discreteIntersectionDesc);
        intersectionMethod = dynamic_cast<Intersection*>(obj.get());
    }
}

void PipelineImpl::reset()
{
    computeCollisionReset();
}

void PipelineImpl::computeCollisionReset()
{
    const simulation::Node* root = dynamic_cast<simulation::Node*>(getContext());
    if(root == nullptr) return;
    if (broadPhaseDetection!=nullptr && broadPhaseDetection->getIntersectionMethod()!=intersectionMethod)
        broadPhaseDetection->setIntersectionMethod(intersectionMethod);
    if (narrowPhaseDetection!=nullptr && narrowPhaseDetection->getIntersectionMethod()!=intersectionMethod)
        narrowPhaseDetection->setIntersectionMethod(intersectionMethod);
    if (contactManager!=nullptr && contactManager->getIntersectionMethod()!=intersectionMethod)
        contactManager->setIntersectionMethod(intersectionMethod);

    SCOPED_TIMER("CollisionReset");
    doCollisionReset();
}

void PipelineImpl::computeCollisionDetection()
{
    simulation::Node* root = dynamic_cast<simulation::Node*>(getContext());
    if(root == nullptr) return;
    SCOPED_TIMER("CollisionDetection");
    std::vector<CollisionModel*> collisionModels;
    root->getTreeObjects<CollisionModel>(&collisionModels);
    doCollisionDetection(collisionModels);
}

void PipelineImpl::computeCollisionResponse()
{
    const simulation::Node* root = dynamic_cast<simulation::Node*>(getContext());
    if(root == nullptr) return;

    SCOPED_TIMER("CollisionResponse");
    doCollisionResponse();
}

} // namespace simulation

} // namespace sofa
