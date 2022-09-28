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

#include <sofa/component/collision/detection/algorithm/BVHNarrowPhase.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/component/collision/detection/algorithm/MirrorIntersector.h>
#include <sofa/helper/ScopedAdvancedTimer.h>

namespace sofa::component::collision::detection::algorithm
{

int BVHNarrowPhaseClass = core::RegisterObject("Narrow phase collision detection based on boundary volume hierarchy")
        .add< BVHNarrowPhase >()
;

BVHNarrowPhase::BVHNarrowPhase() : core::collision::NarrowPhaseDetection()
{}


bool BVHNarrowPhase::isSelfCollision(core::CollisionModel* cm1, core::CollisionModel* cm2)
{
    return (cm1->getContext() == cm2->getContext());
}

void BVHNarrowPhase::addCollisionPair(const std::pair<core::CollisionModel*, core::CollisionModel*>& cmPair)
{
    core::CollisionModel *cm1 = cmPair.first; //->getNext();
    core::CollisionModel *cm2 = cmPair.second; //->getNext();

    if (!cm1->isSimulated() && !cm2->isSimulated())
        return;

    if (cm1->empty() || cm2->empty())
        return;

    core::CollisionModel *finestCollisionModel1 = cm1->getLast();//get the finest CollisionModel which is not a CubeModel
    core::CollisionModel *finestCollisionModel2 = cm2->getLast();

    const bool selfCollision = isSelfCollision(finestCollisionModel1, finestCollisionModel2);

    bool swapModels = false;
    core::collision::ElementIntersector* finestIntersector = intersectionMethod->findIntersector(finestCollisionModel1, finestCollisionModel2, swapModels);//find the method for the finest CollisionModels
    if (finestIntersector == nullptr)
        return;
    if (swapModels)
    {
        std::swap(cm1, cm2);
        std::swap(finestCollisionModel1, finestCollisionModel2);
    }

    sofa::core::collision::DetectionOutputVector*& outputs = this->getDetectionOutputs(finestCollisionModel1, finestCollisionModel2);

    finestIntersector->beginIntersect(finestCollisionModel1, finestCollisionModel2, outputs);//creates outputs if null

    if (finestCollisionModel1 == cm1 || finestCollisionModel2 == cm2)
    {
        // The last model also contains the root element -> it does not only contains the final level of the tree
        finestCollisionModel1 = nullptr;
        finestCollisionModel2 = nullptr;
        finestIntersector = nullptr;
    }

    // Queue used for the iterative form of a tree traversal, avoiding the recursive form
    std::queue< TestPair > externalCells;
    initializeExternalCells(cm1, cm2, externalCells);

    core::collision::ElementIntersector* intersector = nullptr;
    MirrorIntersector mirror;
    cm1 = nullptr; // force later init of intersector
    cm2 = nullptr;

    while (!externalCells.empty())
    {
        TestPair root = externalCells.front();
        externalCells.pop();

        processExternalCell(root,
                            cm1, cm2,
                            intersector,
                            {finestCollisionModel1, finestCollisionModel2, finestIntersector, selfCollision},
                            &mirror, externalCells, outputs);
    }
}

void BVHNarrowPhase::initializeExternalCells(
        core::CollisionModel *cm1,
        core::CollisionModel *cm2,
        std::queue<TestPair>& externalCells)
{
    //See CollisionModel::getInternalChildren(Index), CollisionModel::getExternalChildren(Index) and definition of CollisionModel class
    const CollisionIteratorRange internalChildren1 = cm1->begin().getInternalChildren();
    const CollisionIteratorRange internalChildren2 = cm2->begin().getInternalChildren();
    const CollisionIteratorRange externalChildren1 = cm1->begin().getExternalChildren();
    const CollisionIteratorRange externalChildren2 = cm2->begin().getExternalChildren();

    const auto addToExternalCells = [&externalCells](
            const CollisionIteratorRange& children1,
            const CollisionIteratorRange& children2)
    {
        if (!isRangeEmpty(children1) && !isRangeEmpty(children2))
        {
            externalCells.emplace(children1,children2);
        }
    };

    addToExternalCells(internalChildren1, internalChildren2);
    addToExternalCells(internalChildren1, externalChildren2);
    addToExternalCells(externalChildren1, internalChildren2);
    addToExternalCells(externalChildren1, externalChildren2);
}

void BVHNarrowPhase::processExternalCell(const TestPair &externalCell,
                                              core::CollisionModel *&cm1,
                                              core::CollisionModel *&cm2,
                                              core::collision::ElementIntersector *coarseIntersector,
                                              const FinestCollision &finest,
                                              MirrorIntersector *mirror,
                                              std::queue<TestPair> &externalCells,
                                              sofa::core::collision::DetectionOutputVector *&outputs) const
{
    const auto [collisionModel1, collisionModel2] = getCollisionModelsFromTestPair(externalCell);

    if (cm1 != collisionModel1 || cm2 != collisionModel2)//if the CollisionElements do not belong to cm1 and cm2, update cm1 and cm2
    {
        cm1 = collisionModel1;
        cm2 = collisionModel2;
        if (!cm1 || !cm2) return;

        bool swapModels = false;
        coarseIntersector = intersectionMethod->findIntersector(cm1, cm2, swapModels);

        if (coarseIntersector == nullptr)
        {
            msg_error() << "Unable to find coarseIntersector " << intersectionMethod->getName() << " for "<<cm1->getClassName()<<" - "<<cm2->getClassName();
        }

        if (swapModels)
        {
            mirror->intersector = coarseIntersector;
            coarseIntersector = mirror;
        }
    }

    if (coarseIntersector == nullptr)
        return;

    // Stack used for the iterative form of a tree traversal, avoiding the recursive form
    std::stack< TestPair > internalCells;
    internalCells.push(externalCell);

    while (!internalCells.empty())
    {
        TestPair current = internalCells.top();
        internalCells.pop();

        processInternalCell(current, coarseIntersector, finest, externalCells, internalCells, outputs);
    }
}

void BVHNarrowPhase::processInternalCell(const TestPair &internalCell,
                                              core::collision::ElementIntersector *coarseIntersector,
                                              const FinestCollision &finest,
                                              std::queue<TestPair> &externalCells,
                                              std::stack<TestPair> &internalCells,
                                              sofa::core::collision::DetectionOutputVector *&outputs)
{
    const auto [collisionModel1, collisionModel2] = getCollisionModelsFromTestPair(internalCell);

    if (collisionModel1 == finest.cm1 && collisionModel2 == finest.cm2) //the collision models are the finest ones
    {
        // Final collision pairs
        finalCollisionPairs(internalCell, finest.selfCollision, coarseIntersector, outputs);
    }
    else
    {
        visitCollisionElements(internalCell, coarseIntersector, finest, externalCells, internalCells, outputs);
    }
}

void BVHNarrowPhase::visitCollisionElements(const TestPair &root,
                                                 core::collision::ElementIntersector *coarseIntersector,
                                                 const FinestCollision &finest,
                                                 std::queue<TestPair> &externalCells,
                                                 std::stack<TestPair> &internalCells,
                                                 sofa::core::collision::DetectionOutputVector *&outputs)
{
    const core::CollisionElementIterator begin1 = root.first.first;
    const core::CollisionElementIterator end1 = root.first.second;
    const core::CollisionElementIterator begin2 = root.second.first;
    const core::CollisionElementIterator end2 = root.second.second;

    for (auto it1 = begin1; it1 != end1; ++it1)
    {
        for (auto it2 = begin2; it2 != end2; ++it2)
        {
            if (coarseIntersector->canIntersect(it1, it2))
            {
                // Need to test recursively
                // Note that an element cannot have both internal and external children

                TestPair newInternalTests(it1.getInternalChildren(), it2.getInternalChildren());

                if (!isRangeEmpty(newInternalTests.first))
                {
                    if (!isRangeEmpty(newInternalTests.second))
                    {
                        //both collision elements have internal children. They are added to the list
                        internalCells.push(std::move(newInternalTests));
                    }
                    else
                    {
                        //only the first collision element has internal children. The second collision element
                        //is kept as it is
                        newInternalTests.second = {it2, it2 + 1};
                        internalCells.push(std::move(newInternalTests));
                    }
                }
                else
                {
                    if (!isRangeEmpty(newInternalTests.second))
                    {
                        //only the second collision element has internal children. The first collision element
                        //is kept as it is
                        newInternalTests.first = {it1, it1 + 1};
                        internalCells.push(std::move(newInternalTests));
                    }
                    else
                    {
                        // end of both internal tree of elements.
                        // need to test external children
                        visitExternalChildren(it1, it2, coarseIntersector, finest, externalCells, outputs);
                    }
                }
            }
        }
    }
}

void BVHNarrowPhase::visitExternalChildren(const core::CollisionElementIterator &it1,
                                                const core::CollisionElementIterator &it2,
                                                core::collision::ElementIntersector *coarseIntersector,
                                                const FinestCollision &finest,
                                                std::queue<TestPair> &externalCells,
                                                sofa::core::collision::DetectionOutputVector *&outputs)
{
    const TestPair externalChildren(it1.getExternalChildren(), it2.getExternalChildren());

    const bool isExtChildrenRangeEmpty1 = isRangeEmpty(externalChildren.first);
    const bool isExtChildrenRangeEmpty2 = isRangeEmpty(externalChildren.second);

    if (!isExtChildrenRangeEmpty1)
    {
        if (!isExtChildrenRangeEmpty2)
        {
            const auto [collisionModel1, collisionModel2] = getCollisionModelsFromTestPair(externalChildren);
            if (collisionModel1 == finest.cm1 && collisionModel2 == finest.cm2) //the collision models are the finest ones
            {
                finalCollisionPairs(externalChildren, finest.selfCollision, finest.intersector, outputs);
            }
            else
            {
                externalCells.push(externalChildren);
            }
        }
        else
        {
            // only first element has external children
            // test them against the second element
            externalCells.emplace(externalChildren.first, std::make_pair(it2, it2 + 1));
        }
    }
    else if (!isExtChildrenRangeEmpty2)
    {
        // only second element has external children
        // test them against the first element
        externalCells.emplace(std::make_pair(it1, it1 + 1), externalChildren.second);
    }
    else
    {
        // No child -> final collision pair
        if (!finest.selfCollision || it1.canCollideWith(it2))
            coarseIntersector->intersect(it1, it2, outputs);
    }
}

void BVHNarrowPhase::finalCollisionPairs(const TestPair& pair,
                                              bool selfCollision,
                                              core::collision::ElementIntersector* intersector,
                                              sofa::core::collision::DetectionOutputVector*& outputs)
{
    const core::CollisionElementIterator begin1 = pair.first.first;
    const core::CollisionElementIterator end1 = pair.first.second;
    const core::CollisionElementIterator begin2 = pair.second.first;
    const core::CollisionElementIterator end2 = pair.second.second;

    for (auto it1 = begin1; it1 != end1; ++it1)
    {
        for (auto it2 = begin2; it2 != end2; ++it2)
        {
            // Final collision pair
            if (!selfCollision || it1.canCollideWith(it2))
                intersector->intersect(it1, it2, outputs);
        }
    }
}

std::pair<core::CollisionModel*, core::CollisionModel*> BVHNarrowPhase::getCollisionModelsFromTestPair(const TestPair& pair)
{
    auto* collisionModel1 = pair.first.first.getCollisionModel(); //get the first collision model
    auto* collisionModel2 = pair.second.first.getCollisionModel(); //get the second collision model
    return {collisionModel1, collisionModel2};
}

bool BVHNarrowPhase::isRangeEmpty(const CollisionIteratorRange& range)
{
    return range.first == range.second;
}
} //namespace sofa::component::collision::detection::algorithm
