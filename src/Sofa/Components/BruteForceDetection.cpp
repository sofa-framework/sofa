#include "BruteForceDetection.h"
#include "Sphere.h"
#include "Triangle.h"
#include "Line.h"
#include "Point.h"
#include "Common/FnDispatcher.h"
#include "Common/ObjectFactory.h"
#include "Graph/GNode.h"

#include <map>

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
        bool collisionDetected = false;
        const std::vector<CollisionElement*>& vectElems1 = cm->getCollisionElements();
        const std::vector<CollisionElement*>& vectElems2 = cm2->getCollisionElements();
        for (unsigned int i=0; i<vectElems1.size(); i++)
        {
            for (unsigned int j=0; j<vectElems2.size(); j++)
            {
                if (!vectElems1[i]->canCollideWith(vectElems2[j])) continue;
                if (intersectionMethod->canIntersect(vectElems1[i],vectElems2[j]))
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
    CollisionModel *cm1 = cmPair.first->getNext();
    CollisionModel *cm2 = cmPair.second->getNext();

    if (cm1->isStatic() && cm2->isStatic())
        return;

    const std::vector<CollisionElement*>& vectElems1 = cm1->getCollisionElements();
    const std::vector<CollisionElement*>& vectElems2 = cm2->getCollisionElements();

    if (vectElems1.size()<=0 ||vectElems2.size()<=0)
        return;

    if (!intersectionMethod->isSupported(vectElems1[0], vectElems2[0]))
        return;

    Graph::GNode* node = dynamic_cast<Graph::GNode*>(getContext());
    if (node && !node->getLogTime()) node=NULL; // Only use node for time logging
    Graph::GNode::ctime_t t0=0, t=0;

    const bool continuous = intersectionMethod->useContinuous();
    const bool proximity  = intersectionMethod->useProximity();
    const double distance = intersectionMethod->getAlarmDistance();
    const double dt       = getContext()->getDt();

    Vector3 minBBox1, maxBBox1;
    Vector3 minBBox2, maxBBox2;
    for (unsigned int i=0; i<vectElems1.size(); i++)
    {
        CollisionElement* e1 = vectElems1[i];
        if (continuous)
            e1->getContinuousBBox(minBBox1, maxBBox1, dt);
        else
            e1->getBBox(minBBox1, maxBBox1);
        if (proximity)
        {
            minBBox1[0] -= distance;
            minBBox1[1] -= distance;
            minBBox1[2] -= distance;
            maxBBox1[0] += distance;
            maxBBox1[1] += distance;
            maxBBox1[2] += distance;
        }
        for (unsigned int j=0; j<vectElems2.size(); j++)
        {
            CollisionElement* e2 = vectElems2[j];
            if (!e1->canCollideWith(e2)) continue;
            if (continuous)
                e2->getContinuousBBox(minBBox2, maxBBox2, dt);
            else
                e2->getBBox(minBBox2, maxBBox2);
            if (minBBox1[0] > maxBBox2[0] || minBBox2[0] > maxBBox1[0]
                || minBBox1[1] > maxBBox2[1] || minBBox2[1] > maxBBox1[1]
                || minBBox1[2] > maxBBox2[2] || minBBox2[2] > maxBBox1[2]) continue;
            if (node) t0 = node->startTime();
            bool b = intersectionMethod->canIntersect(e1,e2);
            if (node) t += node->startTime() - t0;
            if (b)
            {
                elemPairs.push_back(std::make_pair(e1,e2));
            }
        }
    }
    if (node) node->addTime(t, "collision", intersectionMethod, this);
}

void BruteForceDetection::draw()
{
    if (!bDraw) return;

    std::vector<std::pair<CollisionElement*, CollisionElement*> >::iterator it = elemPairs.begin();
    std::vector<std::pair<CollisionElement*, CollisionElement*> >::iterator itEnd = elemPairs.end();

    if (elemPairs.size() >= 1)
    {
        glDisable(GL_LIGHTING);
        glColor3f(1.0, 0.0, 1.0);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glLineWidth(3);
        //std::cout << "Size : " << elemPairs.size() << std::endl;
        for (; it != itEnd; it++)
        {
            //std::cout<<*((*it)->getCollisionElement(0)->getSphere())<<std::endl;
            Sphere *s;
            s = dynamic_cast<Sphere*>(it->first);
            if (s!=NULL) s->draw();
            s = dynamic_cast<Sphere*>(it->second);
            if (s!=NULL) s->draw();
            Triangle *t;
            t = dynamic_cast<Triangle*>(it->first);
            if (t!=NULL) t->draw();
            t = dynamic_cast<Triangle*>(it->second);
            if (t!=NULL) t->draw();
            Line *l;
            l = dynamic_cast<Line*>(it->first);
            if (l!=NULL) l->draw();
            l = dynamic_cast<Line*>(it->second);
            if (l!=NULL) l->draw();
            Point *p;
            p = dynamic_cast<Point*>(it->first);
            if (p!=NULL) p->draw();
            p = dynamic_cast<Point*>(it->second);
            if (p!=NULL) p->draw();
            /* Sphere *sph = (*it)->first->getSphere();
            Sphere *sph1 = (*it)->second->getSphere();
            glPushMatrix();
            glTranslated (sph->center.x, sph->center.y, sph->center.z);
            glutSolidSphere(sph->radius, 10, 10);
            glPopMatrix();
            glPushMatrix();
            glTranslated (sph1->center.x, sph1->center.y, sph1->center.z);
            glutSolidSphere(sph1->radius, 10, 10);
            glPopMatrix(); */
            //(*it)->getCollisionElement(0)->getSphere()->draw();
            //(*it)->getCollisionElement(1)->getSphere()->draw();
        }
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glLineWidth(1);
    }
}

} // namespace Components

} // namespace Sofa
