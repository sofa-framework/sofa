/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#include <sofa/component/collision/DefaultPipeline.h>
#include <sofa/core/CollisionModel.h>
#include <sofa/simulation/tree/GNode.h>
#include <sofa/core/ObjectFactory.h>

#include <GL/gl.h>

#define VERBOSE(a) if (bVerbose.getValue()) a; else

namespace sofa
{

namespace component
{

namespace collision
{

using namespace core;
using namespace core::objectmodel;
using namespace core::componentmodel::collision;
using namespace sofa::defaulttype;

SOFA_DECL_CLASS(DefaultPipeline)

int DefaultPipelineClass = core::RegisterObject("The default collision detection and modeling pipeline")
        .add< DefaultPipeline >()
        .addAlias("CollisionPipeline")
        ;

DefaultPipeline::DefaultPipeline()
    : bVerbose(dataField(&bVerbose, false, "verbose","Display current step information"))
    , bDraw(dataField(&bDraw, false, "draw","Draw detected collisions"))
    , depth(dataField(&depth, 6, "depth","Max depth of bounding trees"))
{
}

typedef simulation::tree::GNode::ctime_t ctime_t;

void DefaultPipeline::doCollisionReset()
{
    core::objectmodel::BaseContext* scene = getContext();
    simulation::tree::GNode* node = dynamic_cast<simulation::tree::GNode*>(scene);
    if (node && !node->getLogTime()) node=NULL; // Only use node for time logging
    ctime_t t0 = 0;
    const std::string category = "collision";

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
        if (node) t0 = node->startTime();
        groupManager->clearGroups(scene);
        if (node) t0 = node->endTime(t0, category, groupManager, this);
    }
}

void DefaultPipeline::doCollisionDetection(const std::vector<core::CollisionModel*>& collisionModels)
{
    core::objectmodel::BaseContext* scene = getContext();
    simulation::tree::GNode* node = dynamic_cast<simulation::tree::GNode*>(scene);
    if (node && !node->getLogTime()) node=NULL; // Only use node for time logging
    ctime_t t0 = 0;
    const std::string category = "collision";

    // clear all detection outputs
    //{
    //std::vector< DetectionOutput* >::iterator it = detectionOutputs.begin();
    //std::vector< DetectionOutput* >::iterator itEnd = detectionOutputs.end();
    //for (; it != itEnd; it++)
    //	delete *it;
    //}
    //detectionOutputs.clear();

    VERBOSE(std::cout << "Compute Bounding Trees"<<std::endl);
    // First, we compute a bounding volume for the collision model (for example bounding sphere)
    // or we have loaded a collision model that knows its other model

    std::vector<CollisionModel*> vectBoundingVolume;
    {
        if (node) t0 = node->startTime();
        const bool continuous = intersectionMethod->useContinuous();
        const double dt       = getContext()->getDt();

        std::vector<CollisionModel*>::const_iterator it = collisionModels.begin();
        std::vector<CollisionModel*>::const_iterator itEnd = collisionModels.end();
        int nActive = 0;
        for (; it != itEnd; it++)
        {
            if (!(*it)->isActive()) continue;
            if (continuous)
                (*it)->computeContinuousBoundingTree(dt, depth.getValue());
            else
                (*it)->computeBoundingTree(depth.getValue());
            vectBoundingVolume.push_back ((*it)->getFirst());
            ++nActive;
        }
        if (node) t0 = node->endTime(t0, "collision/bbox", this);
        VERBOSE(std::cout << "Computed "<<nActive<<" BBoxs"<<std::endl);
    }
    // then we start the broad phase
    if (broadPhaseDetection==NULL) return; // can't go further
    VERBOSE(std::cout << "BroadPhaseDetection "<<broadPhaseDetection->getName()<<std::endl);
    if (node) t0 = node->startTime();
    broadPhaseDetection->beginBroadPhase();
    broadPhaseDetection->addCollisionModels(vectBoundingVolume);  // detection is done there
    broadPhaseDetection->endBroadPhase();
    if (node) t0 = node->endTime(t0, category, broadPhaseDetection, this);

    // then we start the narrow phase
    if (narrowPhaseDetection==NULL) return; // can't go further
    VERBOSE(std::cout << "NarrowPhaseDetection "<<narrowPhaseDetection->getName()<<std::endl);
    if (node) t0 = node->startTime();
    narrowPhaseDetection->beginNarrowPhase();
    std::vector<std::pair<CollisionModel*, CollisionModel*> >& vectCMPair = broadPhaseDetection->getCollisionModelPairs();
    VERBOSE(std::cout << vectCMPair.size()<<" colliding model pairs"<<std::endl);
    narrowPhaseDetection->addCollisionPairs(vectCMPair);
    narrowPhaseDetection->endNarrowPhase();
    if (node) t0 = node->endTime(t0, category, narrowPhaseDetection, this);
#if 0 // no longer required as it is done within the narrow phase now
    VERBOSE(std::cout << "CollisionDetection "<<std::endl);
    // then we start the real detection between primitives
    {
        std::vector<std::pair<CollisionElementIterator, CollisionElementIterator> >& vectElemPair = narrowPhaseDetection->getCollisionElementPairs();
        std::vector<std::pair<CollisionElementIterator, CollisionElementIterator> >::iterator it4 = vectElemPair.begin();
        std::vector<std::pair<CollisionElementIterator, CollisionElementIterator> >::iterator it4End = vectElemPair.end();

        // Cache the intersector used
        ElementIntersector* intersector = NULL;
        core::CollisionModel* model1 = NULL;
        core::CollisionModel* model2 = NULL;

        if (node) t0 = node->startTime();
        for (; it4 != it4End; it4++)
        {
            CollisionElementIterator cm1 = it4->first;
            CollisionElementIterator cm2 = it4->second;
            if (cm1.getCollisionModel() != model1 || cm2.getCollisionModel() != model2)
            {
                model1 = cm1.getCollisionModel();
                model2 = cm2.getCollisionModel();
                intersector = intersectionMethod->findIntersector(model1, model2);
            }
            if (intersector != NULL)
            {
                DetectionOutput *detection = intersector->intersect(cm1, cm2);

                if (detection)
                    detectionOutputs.push_back(detection);
            }
        }
        if (node) t0 = node->endTime(t0, category, intersectionMethod, this);
    }
    VERBOSE(std::cout << detectionOutputs.size()<<" collisions detected"<<std::endl);
#endif
}

void DefaultPipeline::doCollisionResponse()
{
    core::objectmodel::BaseContext* scene = getContext();
    simulation::tree::GNode* node = dynamic_cast<simulation::tree::GNode*>(scene);
    if (node && !node->getLogTime()) node=NULL; // Only use node for time logging
    ctime_t t0 = 0;
    const std::string category = "collision";

    // then we start the creation of contacts
    if (contactManager==NULL) return; // can't go further
    VERBOSE(std::cout << "Create Contacts "<<contactManager->getName()<<std::endl);
    if (node) t0 = node->startTime();
    contactManager->createContacts(narrowPhaseDetection->getDetectionOutputs());
    if (node) t0 = node->endTime(t0, category, contactManager, this);

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
        if (node) t0 = node->startTime();
        groupManager->createGroups(scene, notStaticContacts);
        if (node) t0 = node->endTime(t0, category, groupManager, this);
    }
}

void DefaultPipeline::draw()
{
    if (!bDraw.getValue()) return;
    if (!narrowPhaseDetection) return;
    glDisable(GL_LIGHTING);
    glLineWidth(2);
    glBegin(GL_LINES);
    DetectionOutputMap& outputsMap = narrowPhaseDetection->getDetectionOutputs();
    for (DetectionOutputMap::iterator it = outputsMap.begin(); it!=outputsMap.end(); it++)
    {
        DetectionOutputVector& outputs = it->second;
        for (DetectionOutputVector::iterator it2 = outputs.begin(); it2!=outputs.end(); it2++)
        {
            DetectionOutput* d = &*it2;
            if (d->distance<0)
                glColor3f(1.0f,0.5f,0.5f);
            else
                glColor3f(0.5f,1.0f,0.5f);
            glVertex3dv(d->point[0].ptr());
            glVertex3dv(d->point[1].ptr());
        }
    }
    glEnd();
    glLineWidth(1);
}
} // namespace collision

} // namespace component

} // namespace sofa

