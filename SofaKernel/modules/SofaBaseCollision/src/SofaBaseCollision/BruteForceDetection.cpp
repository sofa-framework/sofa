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
#include <SofaBaseCollision/BruteForceDetection.h>

#include <SofaBaseCollision/MirrorIntersector.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/ScopedAdvancedTimer.h>

namespace sofa::component::collision
{

using namespace sofa::defaulttype;
using namespace sofa::helper;
using namespace collision;

int BruteForceDetectionClass = core::RegisterObject("Collision detection using extensive pair-wise tests")
        .add< BruteForceDetection >()
        ;

using namespace core::objectmodel;

BruteForceDetection::BruteForceDetection()
    : box(initData(&box, "box", "if not empty, objects that do not intersect this bounding-box will be ignored"))
{
}

void BruteForceDetection::init()
{
    reinit();
}

void BruteForceDetection::reinit()
{
    if (box.getValue()[0][0] >= box.getValue()[1][0])
    {
        boxModel.reset();
    }
    else
    {
        if (!boxModel) boxModel = sofa::core::objectmodel::New<CubeCollisionModel>();
        boxModel->resize(1);
        boxModel->setParentOf(0, box.getValue()[0], box.getValue()[1]);
    }
}

void BruteForceDetection::addCollisionModel(core::CollisionModel *cm)
{
    if (cm == nullptr || cm->empty())
        return;
	assert(intersectionMethod != nullptr);

    // If a box is defined, check that both collision models are inside the box
    // If both models are outside, ignore them
    if (boxModel)
    {
        bool swapModels = false;
        core::collision::ElementIntersector* intersector = intersectionMethod->findIntersector(cm, boxModel.get(), swapModels);
        if (intersector)
        {
            core::CollisionModel* cm1 = (swapModels?boxModel.get():cm);
            core::CollisionModel* cm2 = (swapModels?cm:boxModel.get());

            // Here we assume a single root element is present in both models
            if (!intersector->canIntersect(cm1->begin(), cm2->begin()))
                return;
        }
    }

    if (cm->isSimulated() && cm->getLast()->canCollideWith(cm->getLast()))
    {
        // self collision
        bool swapModels = false;
        core::collision::ElementIntersector* intersector = intersectionMethod->findIntersector(cm, cm, swapModels);
        if (intersector != nullptr)
            if (intersector->canIntersect(cm->begin(), cm->begin()))
            {
                cmPairs.push_back(std::make_pair(cm, cm));
            }
    }

    // Browse all other collision models to check if there is a potential collision (conservative check)
    for (sofa::helper::vector<core::CollisionModel*>::iterator it = collisionModels.begin(); it != collisionModels.end(); ++it)
    {
        core::CollisionModel* cm2 = *it;

        // ignore this pair if both are NOT simulated (inactive)
        if (!cm->isSimulated() && !cm2->isSimulated())
        {
            continue;
        }

        if (!keepCollisionBetween(cm->getLast(), cm2->getLast()))
            continue;

        bool swapModels = false;
        core::collision::ElementIntersector* intersector = intersectionMethod->findIntersector(cm, cm2, swapModels);
        if (intersector == nullptr)
            continue;

        core::CollisionModel* cm1 = (swapModels?cm2:cm);
        cm2 = (swapModels?cm:cm2);

        // Here we assume a single root element is present in both models
        if (intersector->canIntersect(cm1->begin(), cm2->begin()))
        {
            cmPairs.push_back(std::make_pair(cm1, cm2));
        }
    }
    collisionModels.push_back(cm);
}


bool BruteForceDetection::keepCollisionBetween(core::CollisionModel *cm1, core::CollisionModel *cm2)
{
	return cm1->canCollideWith(cm2) && cm2->canCollideWith(cm1);
}

void BruteForceDetection::addCollisionPair(const std::pair<core::CollisionModel*, core::CollisionModel*>& cmPair)
{
    core::CollisionModel *cm1 = cmPair.first; //->getNext();
    core::CollisionModel *cm2 = cmPair.second; //->getNext();

    if (!cm1->isSimulated() && !cm2->isSimulated())
        return;

    if (cm1->empty() || cm2->empty())
        return;

    core::CollisionModel *finalcm1 = cm1->getLast();//get the finnest CollisionModel which is not a CubeModel
    core::CollisionModel *finalcm2 = cm2->getLast();

    const std::string timerName = "BruteForceDetection addCollisionPair: " + finalcm1->getName() + " - " + finalcm2->getName();
    sofa::helper::ScopedAdvancedTimer bfTimer(timerName);

    bool swapModels = false;
    core::collision::ElementIntersector* finalintersector = intersectionMethod->findIntersector(finalcm1, finalcm2, swapModels);//find the method for the finnest CollisionModels
    if (finalintersector == nullptr)
        return;
    if (swapModels)
    {
        std::swap(cm1, cm2);
        std::swap(finalcm1, finalcm2);
    }

    // Self collision: do both collision elements belong to the same object?
    const bool selfCollision = (finalcm1->getContext() == finalcm2->getContext());

    sofa::core::collision::DetectionOutputVector*& outputs = this->getDetectionOutputs(finalcm1, finalcm2);

    finalintersector->beginIntersect(finalcm1, finalcm2, outputs);//creates outputs if null

    if (finalcm1 == cm1 || finalcm2 == cm2)
    {
        // The last model also contains the root element -> it does not only contains the final level of the tree
        finalcm1 = nullptr;
        finalcm2 = nullptr;
        finalintersector = nullptr;
    }

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

        processExternalCell(root, finalcm1, finalcm2, intersector, finalintersector, &mirror, externalCells, selfCollision, outputs);
    }
}

void BruteForceDetection::initializeExternalCells(
        core::CollisionModel *cm1,
        core::CollisionModel *cm2,
        std::queue<TestPair>& externalCells)
{
    //See CollisionModel::getInternalChildren(Index) and CollisionModel::getExternalChildren(Index)
    const CollisionIteratorRange internalChildren1 = cm1->begin().getInternalChildren();
    const CollisionIteratorRange internalChildren2 = cm2->begin().getInternalChildren();
    const CollisionIteratorRange externalChildren1 = cm1->begin().getExternalChildren();
    const CollisionIteratorRange externalChildren2 = cm2->begin().getExternalChildren();

    const auto isChildrenRangeEmpty = [](const CollisionIteratorRange& children)
    {
        return children.first == children.second;
    };

    const auto addToExternalCells = [&isChildrenRangeEmpty, &externalCells](
            const CollisionIteratorRange& children1,
            const CollisionIteratorRange& children2)
    {
        if (!isChildrenRangeEmpty(children1) && !isChildrenRangeEmpty(children2))
        {
            externalCells.emplace(children1,children2);
        }
    };

    addToExternalCells(internalChildren1, internalChildren2);
    addToExternalCells(internalChildren1, externalChildren2);
    addToExternalCells(externalChildren1, internalChildren2);
    addToExternalCells(externalChildren1, externalChildren2);
}

void BruteForceDetection::processExternalCell(const TestPair& root,
                                              core::CollisionModel *finalcm1,
                                              core::CollisionModel *finalcm2,
                                              core::collision::ElementIntersector* intersector,
                                              core::collision::ElementIntersector* finalintersector,
                                              MirrorIntersector* mirror,
                                              std::queue<TestPair>& externalCells,
                                              bool selfCollision,
                                              sofa::core::collision::DetectionOutputVector*& outputs)
{
    core::CollisionModel *cm1 {nullptr};
    core::CollisionModel *cm2 {nullptr};

    const auto& range1 = root.first;
    const auto& range2 = root.second;

    if (cm1 != range1.first.getCollisionModel() || cm2 != range2.first.getCollisionModel())//if the CollisionElements do not belong to cm1 and cm2, update cm1 and cm2
    {
        cm1 = range1.first.getCollisionModel();
        cm2 = range2.first.getCollisionModel();
        if (!cm1 || !cm2) return;

        bool swapModels = false;
        intersector = intersectionMethod->findIntersector(cm1, cm2, swapModels);

        if (intersector == nullptr)
        {
            msg_error() << "BruteForceDetection: Error finding intersector " << intersectionMethod->getName() << " for "<<cm1->getClassName()<<" - "<<cm2->getClassName()<<sendl;
        }

        if (swapModels)
        {
            mirror->intersector = intersector;
            intersector = mirror;
        }
    }
    if (intersector == nullptr)
        return;
    std::stack< TestPair > internalCells;
    internalCells.push(root);

    while (!internalCells.empty())
    {
        TestPair current = internalCells.top();
        internalCells.pop();

        processInternalCell(current, finalcm1, finalcm2, intersector, finalintersector, externalCells, internalCells, selfCollision, outputs);
    }
}

void BruteForceDetection::processInternalCell(const TestPair& root,
                                              core::CollisionModel *finalcm1,
                                              core::CollisionModel *finalcm2,
                                              core::collision::ElementIntersector* intersector,
                                              core::collision::ElementIntersector* finalintersector,
                                              std::queue<TestPair>& externalCells,
                                              std::stack<TestPair>& internalCells,
                                              bool selfCollision,
                                              sofa::core::collision::DetectionOutputVector*& outputs)
{
    core::CollisionElementIterator begin1 = root.first.first;
    core::CollisionElementIterator end1 = root.first.second;
    core::CollisionElementIterator begin2 = root.second.first;
    core::CollisionElementIterator end2 = root.second.second;

    if (begin1.getCollisionModel() == finalcm1 && begin2.getCollisionModel() == finalcm2)
    {
        // Final collision pairs
        for (core::CollisionElementIterator it1 = begin1; it1 != end1; ++it1)
        {
            for (core::CollisionElementIterator it2 = begin2; it2 != end2; ++it2)
            {
                if (!selfCollision || it1.canCollideWith(it2))
                    intersector->intersect(it1,it2,outputs);
            }
        }
    }
    else
    {
        for (core::CollisionElementIterator it1 = begin1; it1 != end1; ++it1)
        {
            for (core::CollisionElementIterator it2 = begin2; it2 != end2; ++it2)
            {
                //if (self && !it1.canCollideWith(it2)) continue;
                //if (!it1->canCollideWith(it2)) continue;

                bool b = intersector->canIntersect(it1,it2);
                if (b)
                {
                    // Need to test recursively
                    // Note that an element cannot have both internal and external children

                    TestPair newInternalTests(it1.getInternalChildren(),it2.getInternalChildren());
                    TestPair newExternalTests(it1.getExternalChildren(),it2.getExternalChildren());
                    if (newInternalTests.first.first != newInternalTests.first.second)
                    {
                        if (newInternalTests.second.first != newInternalTests.second.second)
                        {
                            internalCells.push(newInternalTests);
                        }
                        else
                        {
                            newInternalTests.second.first = it2;
                            newInternalTests.second.second = it2;
                            ++newInternalTests.second.second;
                            internalCells.push(newInternalTests);
                        }
                    }
                    else
                    {
                        if (newInternalTests.second.first != newInternalTests.second.second)
                        {
                            newInternalTests.first.first = it1;
                            newInternalTests.first.second = it1;
                            ++newInternalTests.first.second;
                            internalCells.push(newInternalTests);
                        }
                        else
                        {
                            // end of both internal tree of elements.
                            // need to test external children
                            if (newExternalTests.first.first != newExternalTests.first.second)
                            {
                                if (newExternalTests.second.first != newExternalTests.second.second)
                                {
                                    if (newExternalTests.first.first.getCollisionModel() == finalcm1 && newExternalTests.second.first.getCollisionModel() == finalcm2)
                                    {
                                        core::CollisionElementIterator extBegin1 = newExternalTests.first.first;
                                        core::CollisionElementIterator extEnd1 = newExternalTests.first.second;
                                        core::CollisionElementIterator extBegin2 = newExternalTests.second.first;
                                        core::CollisionElementIterator extEnd2 = newExternalTests.second.second;
                                        for (core::CollisionElementIterator extIt1 = extBegin1; extIt1 != extEnd1; ++extIt1)
                                        {
                                            for (core::CollisionElementIterator extIt2 = extBegin2; extIt2 != extEnd2; ++extIt2)
                                            {
                                                //if (!extIt1->canCollideWith(extIt2)) continue;
                                                // Final collision pair
                                                if (!selfCollision || extIt1.canCollideWith(extIt2))
                                                    finalintersector->intersect(extIt1,extIt2,outputs);
                                            }
                                        }
                                    }
                                    else
                                        externalCells.push(newExternalTests);
                                }
                                else
                                {
                                    // only first element has external children
                                    // test them against the second element
                                    newExternalTests.second.first = it2;
                                    newExternalTests.second.second = it2;
                                    ++newExternalTests.second.second;
                                    externalCells.emplace(newExternalTests.first, newInternalTests.second);
                                }
                            }
                            else if (newExternalTests.second.first != newExternalTests.second.second)
                            {
                                // only first element has external children
                                // test them against the first element
                                newExternalTests.first.first = it1;
                                newExternalTests.first.second = it1;
                                ++newExternalTests.first.second;
                                externalCells.emplace(newExternalTests.first, newExternalTests.second);
                            }
                            else
                            {
                                // No child -> final collision pair
                                if (!selfCollision || it1.canCollideWith(it2))
                                    intersector->intersect(it1,it2, outputs);
                            }
                        }
                    }
                }
            }
        }
    }
}

} // namespace sofa::component::collision
