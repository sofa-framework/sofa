#include "BruteForceDetection.h"
#include "Sphere.h"
#include "Common/FnDispatcher.h"
#include "Common/ObjectFactory.h"

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
    obj = new BruteForceDetection(arg->getName());
}

SOFA_DECL_CLASS(BruteForce)

Creator<ObjectFactory, BruteForceDetection> BruteForceDetectionClass("BruteForceDetection");

using namespace Abstract;

void BruteForceDetection::addCollisionModel(CollisionModel *cm)
{
    for (std::vector<CollisionModel*>::iterator it = collisionModels.begin(); it != collisionModels.end(); ++it)
    {
        CollisionModel* cm2 = *it;
        bool collisionDetected = false;
        const std::vector<CollisionElement*>& vectElems1 = cm->getCollisionElements();
        const std::vector<CollisionElement*>& vectElems2 = cm2->getCollisionElements();
        for (unsigned int i=0; i<vectElems1.size(); i++)
        {
            for (unsigned int j=0; j<vectElems2.size(); j++)
            {
                if (FnCollisionDetection::getInstance()->intersection(*vectElems1[i],*vectElems2[j]))
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

    const std::vector<CollisionElement*>& vectElems1 = cm1->getCollisionElements();
    const std::vector<CollisionElement*>& vectElems2 = cm2->getCollisionElements();
    for (unsigned int i=0; i<vectElems1.size(); i++)
    {
        CollisionElement* e1 = vectElems1[i];
        for (unsigned int j=0; j<vectElems2.size(); j++)
        {
            CollisionElement* e2 = vectElems2[j];
            if (FnCollisionDetection::getInstance()->intersection(*e1,*e2))
            {
                elemPairs.push_back(std::make_pair(e1,e2));
            }
        }
    }
}

void BruteForceDetection::draw()
{
//	if (!Scene::getInstance()->getShowCollisionModels()) return;

    std::vector<std::pair<CollisionElement*, CollisionElement*> >::iterator it = elemPairs.begin();
    std::vector<std::pair<CollisionElement*, CollisionElement*> >::iterator itEnd = elemPairs.end();

    if (elemPairs.size() >= 1)
    {
        glDisable(GL_LIGHTING);
        glColor3f(1.0, 0.0, 1.0);
        //std::cout << "Size : " << elemPairs.size() << std::endl;
        for (; it != itEnd; it++)
        {
            //std::cout<<*((*it)->getCollisionElement(0)->getSphere())<<std::endl;
            Sphere *s;
            s = dynamic_cast<Sphere*>(it->first);
            if (s!=NULL) s->draw();
            s = dynamic_cast<Sphere*>(it->second);
            if (s!=NULL) s->draw();
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
    }
}

} // namespace Components

} // namespace Sofa
