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
        bool collisionDetected = false;
        //const std::vector<CollisionElement*>& vectElems1 = cm->getCollisionElements();
        //const std::vector<CollisionElement*>& vectElems2 = cm2->getCollisionElements();
        CollisionElementIterator begin1 = cm->begin();
        CollisionElementIterator end1 = cm->end();
        CollisionElementIterator begin2 = cm2->begin();
        CollisionElementIterator end2 = cm2->end();
        for (CollisionElementIterator it1 = begin1; it1 != end1; ++it1)
        {
            for (CollisionElementIterator it2 = begin2; it2 != end2; ++it2)
            {
                //if (!it1->canCollideWith(it2)) continue;
                if (intersector->canIntersect(it1, it2))
                {
                    collisionDetected = true;
                    break;
                }
            }
            if (collisionDetected) break;
        }
        if (collisionDetected)
        {
            cmPairs.push_back(std::make_pair(cm, cm2));
        }
    }
    collisionModels.push_back(cm);
}

void BruteForceDetection::addCollisionPair(const std::pair<CollisionModel*, CollisionModel*>& cmPair)
{
    typedef std::pair< std::pair<CollisionElementIterator,CollisionElementIterator>, std::pair<CollisionElementIterator,CollisionElementIterator> > TestPair;

    CollisionModel *cm1 = cmPair.first->getNext();
    CollisionModel *cm2 = cmPair.second->getNext();

    if (cm1->isStatic() && cm2->isStatic())
        return;

    while (cm1->empty() && cm1->getNext()!=NULL)
        cm1 = cm1->getNext();
    while (cm2->empty() && cm2->getNext()!=NULL)
        cm2 = cm2->getNext();

    if (cm1->empty() || cm2->empty())
        return;

    CollisionModel *finalcm1 = cm1->getLast();
    CollisionModel *finalcm2 = cm2->getLast();
    Collision::ElementIntersector* finalintersector = intersectionMethod->findIntersector(finalcm1, finalcm2);
    if (finalintersector == NULL)
        return;

    Graph::GNode* node = dynamic_cast<Graph::GNode*>(getContext());
    if (node && !node->getLogTime()) node=NULL; // Only use node for time logging
    Graph::GNode::ctime_t t0=0, t=0;
    Graph::GNode::ctime_t ft=0;

    std::queue< TestPair > externalCells;
    externalCells.push(std::make_pair(std::make_pair(cm1->begin(),cm1->end()),std::make_pair(cm2->begin(),cm2->end())));

    Collision::ElementIntersector* intersector = intersectionMethod->findIntersector(cm1, cm2);

    while (!externalCells.empty())
    {
        TestPair root = externalCells.front();
        externalCells.pop();

        if (cm1 != root.first.first.getCollisionModel() || cm2 != root.second.first.getCollisionModel())
        {
            cm1 = root.first.first.getCollisionModel();
            cm2 = root.second.first.getCollisionModel();
            intersector = intersectionMethod->findIntersector(cm1, cm2);
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
                        if (it1->getCollisionModel() == finalcm1 && it2->getCollisionModel() == finalcm2)
                        {
                            // Final collision pair
                            elemPairs.push_back(std::make_pair(it1,it2));
                        }
                        else
                        {
                            // Need to test recursively
                            TestPair newInternalTests(it1->getInternalChildren(),it2->getInternalChildren());
                            TestPair newExternalTests(it1->getExternalChildren(),it2->getExternalChildren());
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
                            }
                            if (newExternalTests.first.first != newExternalTests.first.second && newExternalTests.second.first != newExternalTests.second.second)
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
    if (node)
    {
        std::string name = "collision/";
        name += finalintersector->name();
        node->addTime(ft, name, intersectionMethod);
        t += ft;
        node->addTime(t, "collision", intersectionMethod, this);
    }
}

void BruteForceDetection::draw()
{
    if (!bDraw) return;

    std::vector<std::pair<CollisionElementIterator, CollisionElementIterator> >::iterator it = elemPairs.begin();
    std::vector<std::pair<CollisionElementIterator, CollisionElementIterator> >::iterator itEnd = elemPairs.end();

    if (elemPairs.size() >= 1)
    {
        glDisable(GL_LIGHTING);
        glColor3f(1.0, 0.0, 1.0);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glLineWidth(3);
        //std::cout << "Size : " << elemPairs.size() << std::endl;
        for (; it != itEnd; it++)
        {
            it->first->draw();
            it->second->draw();
        }
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glLineWidth(1);
    }
}

} // namespace Components

} // namespace Sofa
