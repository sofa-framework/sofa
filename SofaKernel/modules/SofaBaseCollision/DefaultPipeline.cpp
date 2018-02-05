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
#include <SofaBaseCollision/DefaultPipeline.h>

#include <sofa/core/CollisionModel.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/collision/BroadPhaseDetection.h>
#include <sofa/core/collision/NarrowPhaseDetection.h>
#include <sofa/core/collision/CollisionGroupManager.h>
#include <sofa/core/collision/ContactManager.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/simulation/Node.h>

#ifdef SOFA_DUMP_VISITOR_INFO
#include <sofa/simulation/Visitor.h>
#endif

#include <sofa/helper/AdvancedTimer.h>
using sofa::helper::AdvancedTimer ;
using sofa::helper::ScopedAdvancedTimer ;


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

SOFA_DECL_CLASS(DefaultPipeline)

int DefaultPipelineClass = core::RegisterObject("The default collision detection and modeling pipeline")
        .add< DefaultPipeline >()
        .addAlias("CollisionPipeline")
        ;

DefaultPipeline::DefaultPipeline()
    : d_doPrintInfoMessage(initData(&d_doPrintInfoMessage, false, "verbose",
                                    "Display extra informations at each computation step. (default=false)"))
    , d_doDebugDraw(initData(&d_doDebugDraw, false, "draw",
                             "Draw the detected collisions. (default=false)"))

    //TODO(dmarchal 2017-05-16) Fix the min & max value with response from a github issue. Remove in 1 year if not done.
    , d_depth(initData(&d_depth, 6, "depth",
                       "Max depth of bounding trees. (default=6, min=?, max=?)"))
{
}

#ifdef SOFA_DUMP_VISITOR_INFO
typedef simulation::Visitor::ctime_t ctime_t;
#endif

void DefaultPipeline::init()
{
    Inherit1::init() ;

    /// Insure that all the value provided by the user are valid and report message if it is not.
    checkDataValues() ;
}

void DefaultPipeline::checkDataValues()
{
    if(d_depth.getValue() < 0)
    {
        msg_warning() << "Invalid value 'depth'="<<d_depth.getValue() << "." << msgendl
                      << "Replaced with the default value = 6." ;
        d_depth.setValue(6) ;
    }
}

void DefaultPipeline::doCollisionReset()
{
    msg_info_when(d_doPrintInfoMessage.getValue())
            << "DefaultPipeline::doCollisionReset" ;

    core::objectmodel::BaseContext* scene = getContext();
    // clear all contacts
    if (contactManager!=nullptr)
    {
        const helper::vector<Contact::SPtr>& contacts = contactManager->getContacts();
        for (helper::vector<Contact::SPtr>::const_iterator it = contacts.begin(); it!=contacts.end(); ++it)
        {
            (*it)->removeResponse();
        }
    }
    // clear all collision groups
    if (groupManager!=nullptr)
    {
        groupManager->clearGroups(scene);
    }
}

void DefaultPipeline::doCollisionDetection(const helper::vector<core::CollisionModel*>& collisionModels)
{
    ScopedAdvancedTimer docollisiontimer("doCollisionDetection");

    msg_info_when(d_doPrintInfoMessage.getValue())
         << "doCollisionDetection, compute Bounding Trees" ;

    // First, we compute a bounding volume for the collision model (for example bounding sphere)
    // or we have loaded a collision model that knows its other model

    helper::vector<CollisionModel*> vectBoundingVolume;
    {
        ScopedAdvancedTimer bboxtimer("BBox");

#ifdef SOFA_DUMP_VISITOR_INFO
        simulation::Visitor::printNode("ComputeBoundingTree");
#endif
        const bool continuous = intersectionMethod->useContinuous();
        const SReal dt       = getContext()->getDt();

        helper::vector<CollisionModel*>::const_iterator it;
        const helper::vector<CollisionModel*>::const_iterator itEnd = collisionModels.end();
        int nActive = 0;

        for (it = collisionModels.begin(); it != itEnd; ++it)
        {
            msg_info_when(d_doPrintInfoMessage.getValue())
                << "doCollisionDetection, consider model" ;

            if (!(*it)->isActive()) continue;

            int used_depth = broadPhaseDetection->needsDeepBoundingTree() ? d_depth.getValue() : 0;

            if (continuous)
                (*it)->computeContinuousBoundingTree(dt, used_depth);
            else
                (*it)->computeBoundingTree(used_depth);

            vectBoundingVolume.push_back ((*it)->getFirst());
            ++nActive;
        }

#ifdef SOFA_DUMP_VISITOR_INFO
        simulation::Visitor::printCloseNode("ComputeBoundingTree");
#endif

        msg_info_when(d_doPrintInfoMessage.getValue())
                << "doCollisionDetection, Computed "<<nActive<<" BBoxs" ;
    }
    // then we start the broad phase
    if (broadPhaseDetection==nullptr) return; // can't go further

    msg_info_when(d_doPrintInfoMessage.getValue())
            << "doCollisionDetection, BroadPhaseDetection "<<broadPhaseDetection->getName();

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("BroadPhase");
#endif
    {
        ScopedAdvancedTimer broadphase("BroadPhase");
        intersectionMethod->beginBroadPhase();
        broadPhaseDetection->beginBroadPhase();
        broadPhaseDetection->addCollisionModels(vectBoundingVolume);  // detection is done there
        broadPhaseDetection->endBroadPhase();
        intersectionMethod->endBroadPhase();
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("BroadPhase");
#endif

    // then we start the narrow phase
    if (narrowPhaseDetection==nullptr) return; // can't go further

    msg_info_when(d_doPrintInfoMessage.getValue())
        << "doCollisionDetection, NarrowPhaseDetection "<<narrowPhaseDetection->getName();

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("NarrowPhase");
#endif
    {
        ScopedAdvancedTimer narrowphase("NarrowPhase");
        intersectionMethod->beginNarrowPhase();
        narrowPhaseDetection->beginNarrowPhase();
        helper::vector<std::pair<CollisionModel*, CollisionModel*> >& vectCMPair = broadPhaseDetection->getCollisionModelPairs();

        msg_info_when(d_doPrintInfoMessage.getValue())
                << "doCollisionDetection, "<< vectCMPair.size()<<" colliding model pairs" ;

        narrowPhaseDetection->addCollisionPairs(vectCMPair);
        narrowPhaseDetection->endNarrowPhase();
        intersectionMethod->endNarrowPhase();
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("NarrowPhase");
#endif

}

void DefaultPipeline::doCollisionResponse()
{
    core::objectmodel::BaseContext* scene = getContext();
    // then we start the creation of contacts
    if (contactManager==nullptr) return; // can't go further

    msg_info_when(d_doPrintInfoMessage.getValue())
        << "Create Contacts "<<contactManager->getName() ;

    contactManager->createContacts(narrowPhaseDetection->getDetectionOutputs());

    // finally we start the creation of collisionGroup

    const helper::vector<Contact::SPtr>& contacts = contactManager->getContacts();

    // First we remove all contacts with non-simulated objects and directly add them
    helper::vector<Contact::SPtr> notStaticContacts;

    for (helper::vector<Contact::SPtr>::const_iterator it = contacts.begin(); it!=contacts.end(); ++it)
    {
        Contact::SPtr c = *it;
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

    if (groupManager==nullptr)
    {
        msg_info_when(d_doPrintInfoMessage.getValue())
            << "Linking all contacts to Scene" ;

        for (helper::vector<Contact::SPtr>::const_iterator it = notStaticContacts.begin(); it!=notStaticContacts.end(); ++it)
        {
            (*it)->createResponse(scene);
        }
    }
    else
    {
        msg_info_when(d_doPrintInfoMessage.getValue())
           << "Create Groups "<<groupManager->getName();

        groupManager->createGroups(scene, notStaticContacts);
    }
}

std::set< std::string > DefaultPipeline::getResponseList() const
{
    std::set< std::string > listResponse;
    core::collision::Contact::Factory::iterator it;
    for (it=core::collision::Contact::Factory::getInstance()->begin(); it!=core::collision::Contact::Factory::getInstance()->end(); ++it)
    {
        listResponse.insert(it->first);
    }
    return listResponse;
}

void DefaultPipeline::draw(const core::visual::VisualParams* )
{
    if (!d_doDebugDraw.getValue()) return;
    if (!narrowPhaseDetection) return;

//TODO(dmarchal 2017-05-17): remove this code or reactivate or do a proper #ifdef
//TODO(dmarchal): it makes also no sense to keep a 'draw' attribute while nothing is displayed.
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

