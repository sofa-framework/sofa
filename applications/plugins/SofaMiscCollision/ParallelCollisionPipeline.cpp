/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaMiscCollision/ParallelCollisionPipeline.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ParallelCollisionModel.h>
#include <sofa/core/collision/ParallelNarrowPhaseDetection.h>
#include <sofa/simulation/Node.h>
#include <sofa/core/ObjectFactory.h>

#include <athapascan-1>

#include <sofa/helper/system/gl.h>

#define VERBOSE(a) if (bVerbose.getValue()) a; else {}

namespace sofa
{

namespace component
{

namespace collision
{

using namespace core;
using namespace core::objectmodel;
using namespace core::collision;
using namespace sofa::defaulttype;

SOFA_DECL_CLASS(ParallelCollisionPipeline)

int ParallelCollisionPipelineClass = core::RegisterObject("A parallel version of the collision detection and modeling pipeline")
        .add< ParallelCollisionPipeline >()
        .addAlias("ParallelPipeline")
        ;

ParallelCollisionPipeline::ParallelCollisionPipeline()
    : bVerbose(initData(&bVerbose, false, "verbose","Display current step information"))
    , bDraw(initData(&bDraw, false, "draw","Draw detected collisions"))
    , depth(initData(&depth, 6, "depth","Max depth of bounding trees"))
    , parallelBoundingTreeDone(0)
{
}

typedef simulation::tree::GVisitor::ctime_t ctime_t;


void ParallelCollisionPipeline::parallelComputeCollisions()
{
    std::cout << "parallelComputeCollisions" << std::endl;
    simulation::Node* root = dynamic_cast<simulation::Node*>(getContext());
    if(root == NULL) return;
    parallelCollisionModels.clear();
    root->getTreeObjects<ParallelCollisionModel>(&parallelCollisionModels);

    while (parallelBoundingTreeDone.size() < parallelCollisionModels.size())
        parallelBoundingTreeDone.push_back(new a1::Shared<bool>(false));

    const bool continuous = intersectionMethod->useContinuous();
    const double dt       = getContext()->getDt();

    std::map<core::CollisionModel*, int> cmMap;
    {
        sofa::helper::vector<ParallelCollisionModel*>::const_iterator it = parallelCollisionModels.begin();
        sofa::helper::vector<ParallelCollisionModel*>::const_iterator itEnd = parallelCollisionModels.end();
        for (int i = 0; it != itEnd; ++it, ++i)
        {
            core::ParallelCollisionModel* cm = *it;
            cmMap[cm] = i;
            if (cm->isMoving() || cm->isSimulated())
                (*it)->computeBoundingTreeParallel(continuous ? dt : 0.0, depth.getValue(), *parallelBoundingTreeDone[i]); //, parallelBoundingTreeDoneAll);
            else
            {
                // compute static models now
                cm->computeBoundingTree(depth.getValue());
            }
        }
    }
    core::collision::ParallelNarrowPhaseDetection* parallelNarrowPhaseDetection = dynamic_cast<core::collision::ParallelNarrowPhaseDetection*>(narrowPhaseDetection);
    if (parallelNarrowPhaseDetection)
    {
        parallelNarrowPhaseDetection->parallelClearOutputs();
        const sofa::helper::vector<std::pair<CollisionModel*, CollisionModel*> >& vectCMPair = broadPhaseDetection->getCollisionModelPairs();
        int np = 0;
        if (!vectCMPair.empty())
        {
            for (unsigned int ip = 0; ip < vectCMPair.size(); ++ip)
            {
                CollisionModel* cm1 = vectCMPair[ip].first->getLast();
                CollisionModel* cm2 = vectCMPair[ip].second->getLast();
                std::map<core::CollisionModel*, int>::const_iterator cm1it = cmMap.find(cm1);
                std::map<core::CollisionModel*, int>::const_iterator cm2it = cmMap.find(cm2);
                if (cm1it == cmMap.end() || cm2it == cmMap.end()) continue;
                int i1 = cm1it->second;
                int i2 = cm2it->second;
                //std::cout << "parallelAddCollisionPair("<<cm1->getTypeName()<<" "<<cm1->getName()<<", "<<cm2->getTypeName()<<" "<<cm2->getName()<<")" << std::endl;
                parallelNarrowPhaseDetection->parallelAddCollisionPair(parallelCollisionModels[i1], parallelCollisionModels[i2], *parallelBoundingTreeDone[i1], *parallelBoundingTreeDone[i2]);
                ++np;
            }
        }
        std::cout << "parallelNarrowPhaseDetection: " << np << " / " << vectCMPair.size() << " collision pairs parallelized." << std::endl;
    }
}

void ParallelCollisionPipeline::doCollisionReset()
{

}
void ParallelCollisionPipeline::doRealCollisionReset()
{
    core::objectmodel::BaseContext* scene = getContext();
    simulation::Node* node = dynamic_cast<simulation::Node*>(scene);
    if (node && !node->getLogTime()) node=NULL; // Only use node for time logging
    ctime_t t0 = 0;
    const std::string category = "collision";

    VERBOSE(sout << "ParallelCollisionPipeline::doCollisionReset, Reset collisions"<<sendl);
    // clear all contacts
    if (contactManager!=NULL)
    {
        const sofa::helper::vector<Contact*>& contacts = contactManager->getContacts();
        for (sofa::helper::vector<Contact*>::const_iterator it = contacts.begin(); it!=contacts.end(); ++it)
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

void ParallelCollisionPipeline::doCollisionDetection(const sofa::helper::vector<core::CollisionModel*>& collisionModels)
{
    //serr<<"ParallelCollisionPipeline::doCollisionDetection"<<sendl;

    core::objectmodel::BaseContext* scene = getContext();
    simulation::Node* node = dynamic_cast<simulation::Node*>(scene);
    if (node && !node->getLogTime()) node=NULL; // Only use node for time logging
    ctime_t t0 = 0;
    const std::string category = "collision";

    VERBOSE(sout << "ParallelCollisionPipeline::doCollisionDetection, Compute Bounding Trees"<<sendl);
    // First, we compute a bounding volume for the collision model (for example bounding sphere)
    // or we have loaded a collision model that knows its other model

    sofa::helper::vector<CollisionModel*> vectBoundingVolume;
    {
        if (node) t0 = node->startTime();
        const bool continuous = intersectionMethod->useContinuous();
        const double dt       = getContext()->getDt();

        sofa::helper::vector<CollisionModel*>::const_iterator it = collisionModels.begin();
        sofa::helper::vector<CollisionModel*>::const_iterator itEnd = collisionModels.end();
        int nActive = 0;
        for (; it != itEnd; ++it)
        {
            VERBOSE(sout << "ParallelCollisionPipeline::doCollisionDetection, consider model "<<(*it)->getName()<<sendl);
            if (!(*it)->isActive()) continue;
            if (!dynamic_cast<core::ParallelCollisionModel*>(*it))
            {
                // we only compute non thread-safe models
                if (continuous)
                    (*it)->computeContinuousBoundingTree(dt, depth.getValue());
                else
                    (*it)->computeBoundingTree(depth.getValue());
            }
            vectBoundingVolume.push_back ((*it)->getFirst());
            ++nActive;
        }
        if (node) t0 = node->endTime(t0, "collision/bbox", this);
        VERBOSE(sout << "ParallelCollisionPipeline::doCollisionDetection, Computed "<<nActive<<" BBoxs"<<sendl);
    }
    // then we start the broad phase
    if (broadPhaseDetection==NULL) return; // can't go further
    VERBOSE(sout << "ParallelCollisionPipeline::doCollisionDetection, BroadPhaseDetection "<<broadPhaseDetection->getName()<<sendl);
    if (node) t0 = node->startTime();
    intersectionMethod->beginBroadPhase();
    broadPhaseDetection->beginBroadPhase();
    broadPhaseDetection->addCollisionModels(vectBoundingVolume);  // detection is done there
    broadPhaseDetection->endBroadPhase();
    intersectionMethod->endBroadPhase();
    if (node) t0 = node->endTime(t0, category, broadPhaseDetection, this);

    // then we start the narrow phase
    if (narrowPhaseDetection==NULL) return; // can't go further
    VERBOSE(sout << "ParallelCollisionPipeline::doCollisionDetection, NarrowPhaseDetection "<<narrowPhaseDetection->getName()<<sendl);
    if (node) t0 = node->startTime();
    intersectionMethod->beginNarrowPhase();
    narrowPhaseDetection->beginNarrowPhase();
    sofa::helper::vector<std::pair<CollisionModel*, CollisionModel*> >& vectCMPair = broadPhaseDetection->getCollisionModelPairs();
    VERBOSE(sout << "ParallelCollisionPipeline::doCollisionDetection, "<< vectCMPair.size()<<" colliding model pairs"<<sendl);
    narrowPhaseDetection->addCollisionPairs(vectCMPair);
    narrowPhaseDetection->endNarrowPhase();
    intersectionMethod->endNarrowPhase();
    if (node) t0 = node->endTime(t0, category, narrowPhaseDetection, this);
}

void ParallelCollisionPipeline::doCollisionResponse()
{
    core::objectmodel::BaseContext* scene = getContext();
    simulation::Node* node = dynamic_cast<simulation::Node*>(scene);
    if (node && !node->getLogTime()) node=NULL; // Only use node for time logging
    ctime_t t0 = 0;
    const std::string category = "collision";

    // then we start the creation of contacts
    if (contactManager==NULL) return; // can't go further
    VERBOSE(sout << "Create Contacts "<<contactManager->getName()<<sendl);
    if (node) t0 = node->startTime();
    contactManager->createContacts(narrowPhaseDetection->getDetectionOutputs());
    if (node) t0 = node->endTime(t0, category, contactManager, this);

    // finally we start the creation of collisionGroup

    const sofa::helper::vector<Contact*>& contacts = contactManager->getContacts();

    // First we remove all contacts with non-simulated objects and directly add them
    sofa::helper::vector<Contact*> notStaticContacts;
    long int c=0;
    for (sofa::helper::vector<Contact*>::const_iterator it = contacts.begin(); it!=contacts.end(); ++it)
    {
        c+=(long int)*it;
    }
    if(c!=contactSum)
    {
        std::cout << "RESET" << std::endl;
        doRealCollisionReset();
        contactSum=c;
    }
    else
    {
        //std::cerr<<"EQUAL!"<<c<<std::endl;
    }
    for (sofa::helper::vector<Contact*>::const_iterator it = contacts.begin(); it!=contacts.end(); ++it)
    {
        Contact* c = *it;
        if (!c->getCollisionModels().first->isSimulated())
        {
            c->createResponse(c->getCollisionModels().second->getContext());
        }
        else if (!c->getCollisionModels().second->isSimulated())
        {
            c->createResponse(c->getCollisionModels().first->getContext());
        }
        else
            notStaticContacts.push_back(c);
    }


    if (groupManager==NULL)
    {
        VERBOSE(sout << "Linking all contacts to Scene"<<sendl);
        for (sofa::helper::vector<Contact*>::const_iterator it = notStaticContacts.begin(); it!=notStaticContacts.end(); ++it)
        {
            (*it)->createResponse(scene);
        }
    }
    else
    {
        VERBOSE(sout << "Create Groups "<<groupManager->getName()<<sendl);
        if (node) t0 = node->startTime();
        groupManager->createGroups(scene, notStaticContacts);
        if (node) t0 = node->endTime(t0, category, groupManager, this);
    }
}

std::set< std::string > ParallelCollisionPipeline::getResponseList() const
{
    std::set< std::string > listResponse;
    core::collision::Contact::Factory::iterator it;
    for (it=core::collision::Contact::Factory::getInstance()->begin(); it!=core::collision::Contact::Factory::getInstance()->end(); ++it)
    {
        listResponse.insert(it->first);
    }
    return listResponse;
}

void ParallelCollisionPipeline::draw(const core::visual::VisualParams* vparams)
{
    if (!bDraw.getValue()) return;
    if (!narrowPhaseDetection) return;
#if 0
    glDisable(GL_LIGHTING);
    glLineWidth(2);
    glBegin(GL_LINES);
    DetectionOutputMap& outputsMap = narrowPhaseDetection->getDetectionOutputs();
    for (DetectionOutputMap::iterator it = outputsMap.begin(); it!=outputsMap.end(); it++)
    {
        /*
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
        */
    }
    glEnd();
    glLineWidth(1);
#endif
}
} // namespace collision

} // namespace component

} // namespace sofa

