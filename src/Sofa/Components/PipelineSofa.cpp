#include "PipelineSofa.h"
#include "Sofa/Abstract/CollisionModel.h"

#include "Common/ObjectFactory.h"

#include <GL/gl.h>

#define VERBOSE(a) if (getVerbose()) a; else

namespace Sofa
{

namespace Components
{

using namespace Abstract;
using namespace Common;
using namespace Collision;

void create(PipelineSofa*& obj, ObjectDescription* arg)
{
    obj = new PipelineSofa();
    obj->setVerbose(atoi(arg->getAttribute("verbose","0"))!=0);
    obj->setDraw(atoi(arg->getAttribute("draw","0"))!=0);
}

SOFA_DECL_CLASS(PipelineSofa)

Creator<ObjectFactory, PipelineSofa> PipelineSofaClass("CollisionPipeline");

PipelineSofa::PipelineSofa()
    : verbose_(false), draw_(false)
{
}

void PipelineSofa::startDetection(const std::vector<Abstract::CollisionModel*>& collisionModels)
{
    Abstract::BaseContext* scene = getContext(); //Scene::getInstance();

    VERBOSE(std::cout << "Reset collisions"<<std::endl);
    // clear all contacts
    if (contactManager!=NULL)
    {
        const std::vector<Contact*>& contacts = contactManager->getContacts();
        for (std::vector<Contact*>::const_iterator it = contacts.begin(); it!=contacts.end(); it++)
        {
            (*it)->removeResponse();
        }
    }
    // clear all collision groups
    if (groupManager!=NULL)
    {
        /// \todo Update for scenegraph
        groupManager->clearGroups(scene);
    }
    // clear all detection outputs
    {
        std::vector< DetectionOutput* >::iterator it = detectionOutputs.begin();
        std::vector< DetectionOutput* >::iterator itEnd = detectionOutputs.end();
        for (; it != itEnd; it++)
            delete *it;
    }
    detectionOutputs.clear();

    VERBOSE(std::cout << "Compute BBoxs"<<std::endl);
    // First, we compute a bounding volume for the collision model (for example bounding sphere)
    // or we have loaded a collision model that knows its other model

    std::vector<CollisionModel*> vectBoundingVolume;
    {
        const bool continuous = intersectionMethod->useContinuous();
        //const double distance = intersectionMethod->getAlarmDistance();
        const double dt       = getContext()->getDt();

        std::vector<CollisionModel*>::const_iterator it = collisionModels.begin();
        std::vector<CollisionModel*>::const_iterator itEnd = collisionModels.end();
        int nActive = 0;
        for (; it != itEnd; it++)
        {
            if (!(*it)->isActive()) continue;
            if (continuous)
                (*it)->computeContinuousBoundingBox(dt);
            else
                (*it)->computeBoundingBox();
            vectBoundingVolume.push_back ((*it)->getFirst());
            ++nActive;
        }
        VERBOSE(std::cout << "Computed "<<nActive<<" BBoxs"<<std::endl);
    }
    // then we start the broad phase
    if (broadPhaseDetection==NULL) return; // can't go further
    VERBOSE(std::cout << "BroadPhaseDetection "<<broadPhaseDetection->getName()<<std::endl);
    broadPhaseDetection->clearBroadPhase();
    broadPhaseDetection->addCollisionModels(vectBoundingVolume);

    // then we start the narrow phase
    if (narrowPhaseDetection==NULL) return; // can't go further
    VERBOSE(std::cout << "NarrowPhaseDetection "<<narrowPhaseDetection->getName()<<std::endl);
    narrowPhaseDetection->clearNarrowPhase();
    std::vector<std::pair<CollisionModel*, CollisionModel*> >& vectCMPair = broadPhaseDetection->getCollisionModelPairs();
    VERBOSE(std::cout << vectCMPair.size()<<" colliding model pairs"<<std::endl);
    narrowPhaseDetection->addCollisionPairs(vectCMPair);

    VERBOSE(std::cout << "CollisionDetection "<<std::endl);
    // then we start the real detection between primitives
    std::vector<std::pair<CollisionElement*, CollisionElement*> >& vectElemPair = narrowPhaseDetection->getCollisionElementPairs();
    std::vector<std::pair<CollisionElement*, CollisionElement*> >::iterator it4 = vectElemPair.begin();
    std::vector<std::pair<CollisionElement*, CollisionElement*> >::iterator it4End = vectElemPair.end();

    for (; it4 != it4End; it4++)
    {
        CollisionElement *cm1 = it4->first;
        CollisionElement *cm2 = it4->second;

        DetectionOutput *detection = intersectionMethod->intersect(cm1, cm2);

        if (detection)
            detectionOutputs.push_back(detection);
    }
    VERBOSE(std::cout << detectionOutputs.size()<<" collisions detected"<<std::endl);


    // then we start the creation of contacts
    if (contactManager==NULL) return; // can't go further
    VERBOSE(std::cout << "Create Contacts "<<contactManager->getName()<<std::endl);
    contactManager->createContacts(detectionOutputs);

    // finally we start the creation of collisionGroup

    const std::vector<Contact*>& contacts = contactManager->getContacts();

    // First we remove all contacts with static objects and directly add them
    std::vector<Contact*> notStaticContacts;

    for (std::vector<Contact*>::const_iterator it = contacts.begin(); it!=contacts.end(); it++)
    {
        Contact* c = *it;
        if (c->getCollisionModels().first->isStatic())
        {
            c->createResponse(c->getCollisionModels().second->getContext());
        }
        else if (c->getCollisionModels().second->isStatic())
        {
            c->createResponse(c->getCollisionModels().first->getContext());
        }
        else
            notStaticContacts.push_back(c);
    }


    if (groupManager==NULL)
    {
        VERBOSE(std::cout << "Linking all contacts to Scene"<<std::endl);
        for (std::vector<Contact*>::const_iterator it = notStaticContacts.begin(); it!=notStaticContacts.end(); it++)
        {
            (*it)->createResponse(scene);
        }
    }
    else
    {
        VERBOSE(std::cout << "Create Groups "<<groupManager->getName()<<std::endl);
        groupManager->createGroups(scene, notStaticContacts);
    }
}

void PipelineSofa::draw()
{
    if (!getDraw()) return;
    glDisable(GL_LIGHTING);
    glLineWidth(2);
    glBegin(GL_LINES);
    std::vector< DetectionOutput* >::iterator it = detectionOutputs.begin();
    std::vector< DetectionOutput* >::iterator itEnd = detectionOutputs.end();
    for (; it != itEnd; it++)
    {
        DetectionOutput* d = *it;
        if (!d->distance)
            glColor3f(1.0f,0.5f,0.5f);
        else
            glColor3f(0.5f,1.0f,0.5f);
        glVertex3dv(d->point[0]);
        glVertex3dv(d->point[1]);
    }
    glEnd();
    glLineWidth(1);
}
} // namespace Components

} // namespace Sofa
