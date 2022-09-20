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

#include <sofa/component/collision/detection/algorithm/BruteForceBroadPhase.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/collision/Intersection.h>

namespace sofa::component::collision::detection::algorithm
{

int BruteForceBroadPhaseClass = core::RegisterObject("Broad phase collision detection using extensive pair-wise tests")
        .add< BruteForceBroadPhase >()
;

BruteForceBroadPhase::BruteForceBroadPhase()
        : box(initData(&box, "box", "if not empty, objects that do not intersect this bounding-box will be ignored"))
{}

void BruteForceBroadPhase::doBaseObjectInit()
{
    reinit();
}

void BruteForceBroadPhase::reinit()
{
    if (box.getValue()[0][0] >= box.getValue()[1][0])
    {
        boxModel.reset();
    }
    else
    {
        if (!boxModel) boxModel = sofa::core::objectmodel::New<collision::geometry::CubeCollisionModel>();
        boxModel->resize(1);
        boxModel->setParentOf(0, box.getValue()[0], box.getValue()[1]);
    }
}

void BruteForceBroadPhase::beginBroadPhase()
{
    core::collision::BroadPhaseDetection::beginBroadPhase();
    m_collisionModels.clear();
}

void BruteForceBroadPhase::addCollisionModel (core::CollisionModel *cm)
{
    if (cm == nullptr || cm->empty())
        return;
    assert(intersectionMethod != nullptr);

    dmsg_info() << "CollisionModel " << cm->getName() << "(" << cm << ") of class " << cm->getClassName()
                << " is added in broad phase (" << m_collisionModels.size() << " collision models)";

    // If a box is defined, check that the collision model intersects the box
    // If the collision model does not intersect the box, it is ignored from the collision detection
    if (boxModel && !intersectWithBoxModel(cm))
    {
        return;
    }

    if (doesSelfCollide(cm))
    {
        // add the collision model to be tested against itself
        cmPairs.emplace_back(cm, cm);
    }

    core::CollisionModel* finalCollisionModel = cm->getLast();

    // Browse all other collision models to check if there is a potential collision (conservative check)
    for (const auto& model : m_collisionModels)
    {
        auto* cm2 = model.firstCollisionModel;
        auto* finalCm2 = model.lastCollisionModel;

        // ignore this pair if both are NOT simulated (inactive)
        if (!cm->isSimulated() && !cm2->isSimulated())
        {
            continue;
        }

        if (!keepCollisionBetween(finalCollisionModel, finalCm2))
            continue;

        bool swapModels = false;
        core::collision::ElementIntersector* intersector = intersectionMethod->findIntersector(cm, cm2, swapModels);
        if (intersector == nullptr)
            continue;

        core::CollisionModel* cm1 = cm;
        if (swapModels)
        {
            std::swap(cm1, cm2);
        }

        // Here we assume a single root element is present in both models
        if (intersector->canIntersect(cm1->begin(), cm2->begin()))
        {
            //both collision models will be further examined in the narrow phase
            cmPairs.emplace_back(cm1, cm2);
        }
    }

    //accumulate CollisionModel's in a vector so the next CollisionModel can be tested against all previous ones
    m_collisionModels.emplace_back(cm, finalCollisionModel);
}

bool BruteForceBroadPhase::keepCollisionBetween(core::CollisionModel *cm1, core::CollisionModel *cm2)
{
    return cm1->canCollideWith(cm2) && cm2->canCollideWith(cm1);
}

bool BruteForceBroadPhase::doesSelfCollide(core::CollisionModel *cm) const
{
    if (cm->isSimulated() && cm->getLast()->canCollideWith(cm->getLast()))
    {
        // self collision
        bool swapModels = false;
        core::collision::ElementIntersector* intersector = intersectionMethod->findIntersector(cm, cm, swapModels);
        if (intersector != nullptr)
        {
            return intersector->canIntersect(cm->begin(), cm->begin());
        }
    }

    return false;
}

bool BruteForceBroadPhase::intersectWithBoxModel(core::CollisionModel *cm) const
{
    bool swapModels = false;
    core::collision::ElementIntersector* intersector = intersectionMethod->findIntersector(cm, boxModel.get(), swapModels);
    if (intersector)
    {
        core::CollisionModel* cm1 = (swapModels?boxModel.get():cm);
        core::CollisionModel* cm2 = (swapModels?cm:boxModel.get());

        // Here we assume a single root element is present in both models
        return intersector->canIntersect(cm1->begin(), cm2->begin());
    }

    return true;
}

}