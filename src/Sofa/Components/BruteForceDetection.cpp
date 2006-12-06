#include "BruteForceDetection.h"
#include "Sphere.h"
#include "Triangle.h"
#include "Line.h"
#include "Point.h"
#include "Common/FnDispatcher.h"
#include "Common/ObjectFactory.h"
#include "Graph/GNode.h"

#include <map>
#include <queue>
#include <stack>

/* for debugging the collision method */
#ifdef _WIN32
#include <windows.h>
#endif
#include <GL/gl.h>
#include <GL/glut.h>

namespace Sofa
{

namespace Components
{

using namespace Common;
using namespace Collision;

void create(BruteForceDetection*& obj, ObjectDescription* arg)
{
    obj = new BruteForceDetection();
    if (arg->getAttribute("draw"))
        obj->setDraw(atoi(arg->getAttribute("draw"))!=0);
}

SOFA_DECL_CLASS(BruteForce)

Creator<ObjectFactory, BruteForceDetection> BruteForceDetectionClass("BruteForceDetection");

using namespace Abstract;

BruteForceDetection::BruteForceDetection()
    : bDraw(false)
{
}

void BruteForceDetection::addCollisionModel(CollisionModel *cm)
{
    if (cm->empty())
        return;
    for (std::vector<CollisionModel*>::iterator it = collisionModels.begin(); it != collisionModels.end(); ++it)
    {
        CollisionModel* cm2 = *it;
        if (cm->isStatic() && cm2->isStatic())
            continue;
        if (!cm->canCollideWith(cm2))
            continue;
        Collision::ElementIntersector* intersector = intersectionMethod->findIntersector(cm, cm2);
        if (intersector == NULL)
            continue;

        // // Here we assume multiple root elements are present in both models
        // bool collisionDetected = false;
        // CollisionElementIterator begin1 = cm->begin();
        // CollisionElementIterator end1 = cm->end();
        // CollisionElementIterator begin2 = cm2->begin();
        // CollisionElementIterator end2 = cm2->end();
        // for (CollisionElementIterator it1 = begin1; it1 != end1; ++it1)
        // {
        //     for (CollisionElementIterator it2 = begin2; it2 != end2; ++it2)
        //     {
        //         //if (!it1->canCollideWith(it2)) continue;
        //         if (intersector->canIntersect(it1, it2))
        //         {
        //             collisionDetected = true;
        //             break;
        //         }
        //     }
        //     if (collisionDetected) break;
        // }
        // if (collisionDetected)

        // Here we assume a single root element is present in both models
        if (intersector->canIntersect(cm->begin(), cm2->begin()))
        {
            //std::cout << "Broad phase "<<cm->getLast()->getName()<<" - "<<cm2->getLast()->getName()<<std::endl;
            cmPairs.push_back(std::make_pair(cm, cm2));
        }
    }
    collisionModels.push_back(cm);
}

void BruteForceDetection::addCollisionPair(const std::pair<CollisionModel*, CollisionModel*>& cmPair)
{
    typedef std::pair< std::pair<CollisionElementIterator,CollisionElementIterator>, std::pair<CollisionElementIterator,CollisionElementIterator> > TestPair;

    CollisionModel *cm1 = cmPair.first; //->getNext();
    CollisionModel *cm2 = cmPair.second; //->getNext();

    //int size0 = elemPairs.size();
    //std::cout << "Narrow phase "<<cm1->getLast()->getName()<<" - "<<cm2->getLast()->getName()<<std::endl;

    if (cm1->isStatic() && cm2->isStatic())
        return;

    if (cm1->empty() || cm2->empty())
        return;

    CollisionModel *finalcm1 = cm1->getLast();
    CollisionModel *finalcm2 = cm2->getLast();
    //std::cout << "Final phase "<<gettypename(typeid(*finalcm1))<<" - "<<gettypename(typeid(*finalcm2))<<std::endl;
    Collision::ElementIntersector* finalintersector = intersectionMethod->findIntersector(finalcm1, finalcm2);
    if (finalintersector == NULL)
        return;
    //std::cout << "Final intersector " << finalintersector->name() << " for "<<finalcm1->getName()<<" - "<<finalcm2->getName()<<std::endl;

    if (finalcm1 == cm1 || finalcm2 == cm2)
    {
        // The last model also contains the root element -> it does not only contains the final level of the tree
        finalcm1 = NULL;
        finalcm2 = NULL;
        finalintersector = NULL;
    }
    Graph::GNode* node = dynamic_cast<Graph::GNode*>(getContext());
    if (node && !node->getLogTime()) node=NULL; // Only use node for time logging
    Graph::GNode::ctime_t t0=0, t=0;
    Graph::GNode::ctime_t ft=0;

    std::queue< TestPair > externalCells;

    std::pair<CollisionElementIterator,CollisionElementIterator> internalChildren1 = cm1->begin().getInternalChildren();
    std::pair<CollisionElementIterator,CollisionElementIterator> internalChildren2 = cm2->begin().getInternalChildren();
    std::pair<CollisionElementIterator,CollisionElementIterator> externalChildren1 = cm1->begin().getExternalChildren();
    std::pair<CollisionElementIterator,CollisionElementIterator> externalChildren2 = cm2->begin().getExternalChildren();
    if (internalChildren1.first != internalChildren1.second)
    {
        if (internalChildren2.first != internalChildren2.second)
            externalCells.push(std::make_pair(internalChildren1,internalChildren2));
        if (externalChildren2.first != externalChildren2.second)
            externalCells.push(std::make_pair(internalChildren1,externalChildren2));
    }
    if (externalChildren1.first != externalChildren1.second)
    {
        if (internalChildren2.first != internalChildren2.second)
            externalCells.push(std::make_pair(externalChildren1,internalChildren2));
        if (externalChildren2.first != externalChildren2.second)
            externalCells.push(std::make_pair(externalChildren1,externalChildren2));
    }
    //externalCells.push(std::make_pair(std::make_pair(cm1->begin(),cm1->end()),std::make_pair(cm2->begin(),cm2->end())));

    //Collision::ElementIntersector* intersector = intersectionMethod->findIntersector(cm1, cm2);
    Collision::ElementIntersector* intersector = NULL;
    cm1 = NULL; // force later init of intersector
    cm2 = NULL;

    while (!externalCells.empty())
    {
        TestPair root = externalCells.front();
        externalCells.pop();

        if (cm1 != root.first.first.getCollisionModel() || cm2 != root.second.first.getCollisionModel())
        {
            cm1 = root.first.first.getCollisionModel();
            cm2 = root.second.first.getCollisionModel();
            intersector = intersectionMethod->findIntersector(cm1, cm2);
            if (intersector == NULL)
            {
                std::cout << "BruteForceDetection: Error finding intersector " << intersectionMethod->getName() << " for "<<gettypename(typeid(*cm1))<<" - "<<gettypename(typeid(*cm2))<<std::endl;
            }
            //else std::cout << "BruteForceDetection: intersector " << intersector->name() << " for " << intersectionMethod->getName() << " for "<<gettypename(typeid(*cm1))<<" - "<<gettypename(typeid(*cm2))<<std::endl;
        }
        if (intersector == NULL)
            continue;
        std::stack< TestPair > internalCells;
        internalCells.push(root);

        Graph::GNode::ctime_t it=0;

        while (!internalCells.empty())
        {
            TestPair current = internalCells.top();
            internalCells.pop();

            CollisionElementIterator begin1 = current.first.first;
            CollisionElementIterator end1 = current.first.second;
            CollisionElementIterator begin2 = current.second.first;
            CollisionElementIterator end2 = current.second.second;
            for (CollisionElementIterator it1 = begin1; it1 != end1; ++it1)
            {
                for (CollisionElementIterator it2 = begin2; it2 != end2; ++it2)
                {
                    //if (!it1->canCollideWith(it2)) continue;

                    //if (node) t0 = node->startTime();
                    bool b = intersector->canIntersect(it1,it2);
                    //if (node) it += node->startTime() - t0;
                    if (b)
                    {
                        if (it1.getCollisionModel() == finalcm1 && it2.getCollisionModel() == finalcm2)
                        {
                            // Final collision pair
                            elemPairs.push_back(std::make_pair(it1,it2));
                        }
                        else
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
                                                CollisionElementIterator begin1 = newExternalTests.first.first;
                                                CollisionElementIterator end1 = newExternalTests.first.second;
                                                CollisionElementIterator begin2 = newExternalTests.second.first;
                                                CollisionElementIterator end2 = newExternalTests.second.second;
                                                for (CollisionElementIterator it1 = begin1; it1 != end1; ++it1)
                                                {
                                                    for (CollisionElementIterator it2 = begin2; it2 != end2; ++it2)
                                                    {
                                                        //if (!it1->canCollideWith(it2)) continue;
                                                        if (node) t0 = node->startTime();
                                                        bool bfinal = finalintersector->canIntersect(it1,it2);
                                                        if (node) ft += node->startTime() - t0;
                                                        // Final collision pair
                                                        if (bfinal)
                                                            elemPairs.push_back(std::make_pair(it1,it2));
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
                                            externalCells.push(std::make_pair(newExternalTests.first, newInternalTests.second));
                                        }
                                    }
                                    else if (newExternalTests.second.first != newExternalTests.second.second)
                                    {
                                        // only first element has external children
                                        // test them against the first element
                                        newExternalTests.first.first = it1;
                                        newExternalTests.first.second = it1;
                                        ++newExternalTests.first.second;
                                        externalCells.push(std::make_pair(newExternalTests.first, newExternalTests.second));
                                    }
                                    else
                                    {
                                        // No child -> final collision pair
                                        elemPairs.push_back(std::make_pair(it1,it2));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if (node)
        {
            //std::string name = "collision/";
            //name += intersector->name();
            //node->addTime(it, name, intersectionMethod);
            t += it;
        }
    }
    if (node && finalintersector!=NULL)
    {
        std::string name = "collision/";
        name += finalintersector->name();
        node->addTime(ft, name, intersectionMethod);
        t += ft;
        node->addTime(t, "collision", intersectionMethod, this);
    }
    //std::cout << "Narrow phase "<<cm1->getLast()->getName()<<"("<<gettypename(typeid(*cm1->getLast()))<<") - "<<cm2->getLast()->getName()<<"("<<gettypename(typeid(*cm2->getLast()))<<"): "<<elemPairs.size()-size0<<" contacts."<<std::endl;
}

void BruteForceDetection::draw()
{
    if (!bDraw) return;

    if (!elemPairs.empty())
    {
        glDisable(GL_LIGHTING);
        glColor3f(1.0, 0.0, 1.0);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glLineWidth(3);
        glPointSize(5);
        std::vector<std::pair<CollisionElementIterator, CollisionElementIterator> >::iterator it = elemPairs.begin();
        std::vector<std::pair<CollisionElementIterator, CollisionElementIterator> >::iterator itEnd = elemPairs.end();
        //std::cout << "Size : " << elemPairs.size() << std::endl;
        for (; it != itEnd; it++)
        {
            it->first.draw();
            it->second.draw();
        }
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glLineWidth(1);
        glPointSize(1);
    }
}

} // namespace Components

} // namespace Sofa
