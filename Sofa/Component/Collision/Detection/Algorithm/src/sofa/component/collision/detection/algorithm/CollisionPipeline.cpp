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
#include <sofa/component/collision/detection/algorithm/CollisionPipeline.h>

#include <sofa/core/CollisionModel.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/collision/BroadPhaseDetection.h>
#include <sofa/core/collision/NarrowPhaseDetection.h>
#include <sofa/core/collision/CollisionGroupManager.h>
#include <sofa/core/collision/ContactManager.h>

#include <sofa/simulation/Node.h>

#ifdef SOFA_DUMP_VISITOR_INFO
#include <sofa/simulation/Visitor.h>
#endif

#include <sofa/helper/ScopedAdvancedTimer.h>
using sofa::helper::ScopedAdvancedTimer ;

#include <sofa/helper/AdvancedTimer.h>


namespace sofa::component::collision::detection::algorithm
{

using namespace core;
using namespace core::objectmodel;
using namespace core::collision;
using namespace sofa::defaulttype;

int CollisionPipelineClass = core::RegisterObject("The default collision detection and modeling pipeline")
        .add< CollisionPipeline >()
        .addAlias("DefaultPipeline")
        ;

const int CollisionPipeline::defaultDepthValue = 6;

CollisionPipeline::CollisionPipeline()
    : d_doPrintInfoMessage(initData(&d_doPrintInfoMessage, false, "verbose",
                                    "Display extra informations at each computation step. (default=false)"))
    , d_doDebugDraw(initData(&d_doDebugDraw, false, "draw",
                             "Draw the detected collisions. (default=false)"))

    //TODO(dmarchal 2017-05-16) Fix the min & max value with response from a github issue. Remove in 1 year if not done.
    , d_depth(initData(&d_depth, defaultDepthValue, "depth",
               ("Max depth of bounding trees. (default=" + std::to_string(defaultDepthValue) + ", min=?, max=?)").c_str()))
{
}

#ifdef SOFA_DUMP_VISITOR_INFO
typedef simulation::Visitor::ctime_t ctime_t;
#endif

void CollisionPipeline::init()
{
    Inherit1::init();

    if (broadPhaseDetection == nullptr)
    {
        msg_warning() << "A BroadPhase component is required to compute collision detection and was not found in the current scene";
    }

    if (narrowPhaseDetection == nullptr)
    {
        msg_warning() << "A NarrowPhase component is required to compute collision detection and was not found in the current scene";
    }

    if (contactManager == nullptr)
    {
        msg_warning() << "A ContactManager component is required to compute collision response and was not found in the current scene";
    }

    /// Insure that all the value provided by the user are valid and report message if it is not.
    checkDataValues() ;
}

void CollisionPipeline::checkDataValues()
{
    if(d_depth.getValue() < 0)
    {
        msg_warning() << "Invalid value 'depth'=" << d_depth.getValue() << "." << msgendl
                      << "Replaced with the default value = " << defaultDepthValue;
        d_depth.setValue(defaultDepthValue) ;
    }
}

void CollisionPipeline::doCollisionReset()
{
    msg_info_when(d_doPrintInfoMessage.getValue())
            << "CollisionPipeline::doCollisionReset" ;

    // clear all contacts
    if (contactManager != nullptr)
    {
        const type::vector<Contact::SPtr>& contacts = contactManager->getContacts();
        for (const auto& contact : contacts)
        {
            if (contact != nullptr)
            {
                contact->removeResponse();
            }
        }
    }

    // clear all collision groups
    if (groupManager != nullptr)
    {
        core::objectmodel::BaseContext* scene = getContext();
        groupManager->clearGroups(scene);
    }
}

void CollisionPipeline::doCollisionDetection(const type::vector<core::CollisionModel*>& collisionModels)
{
    SCOPED_TIMER_VARNAME(docollisiontimer, "doCollisionDetection");

    msg_info_when(d_doPrintInfoMessage.getValue())
         << "doCollisionDetection, compute Bounding Trees" ;

    // First, we compute a bounding volume for the collision model (for example bounding sphere)
    // or we have loaded a collision model that knows its other model

    type::vector<CollisionModel*> vectBoundingVolume;
    {
        SCOPED_TIMER_VARNAME(bboxtimer, "ComputeBoundingTree");

#ifdef SOFA_DUMP_VISITOR_INFO
        simulation::Visitor::printNode("ComputeBoundingTree");
#endif
        const bool continuous = intersectionMethod->useContinuous();
        const SReal dt       = getContext()->getDt();

        type::vector<CollisionModel*>::const_iterator it;
        const type::vector<CollisionModel*>::const_iterator itEnd = collisionModels.end();
        int nActive = 0;

        const int used_depth = (
                    (broadPhaseDetection && broadPhaseDetection->needsDeepBoundingTree()) ||
                    (narrowPhaseDetection && narrowPhaseDetection->needsDeepBoundingTree())
            ) ? d_depth.getValue() : 0;

        for (it = collisionModels.begin(); it != itEnd; ++it)
        {
            msg_info_when(d_doPrintInfoMessage.getValue())
                << "doCollisionDetection, consider model" ;

            if (!(*it)->isActive()) continue;

            if (continuous)
            {
                const std::string msg = "Compute Continuous BoundingTree: " + (*it)->getName();
                ScopedAdvancedTimer continuousBoundingTreeTimer(msg.c_str());
                (*it)->computeContinuousBoundingTree(dt, used_depth);
            }
            else
            {
                std::string msg = "Compute BoundingTree: " + (*it)->getName();
                ScopedAdvancedTimer boundingTreeTimer(msg.c_str());
                (*it)->computeBoundingTree(used_depth);
            }

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
    if (broadPhaseDetection == nullptr)
    {
        return; // can't go further
    }

    msg_info_when(d_doPrintInfoMessage.getValue())
            << "doCollisionDetection, BroadPhaseDetection "<<broadPhaseDetection->getName();

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("BroadPhase");
#endif
    {
        SCOPED_TIMER_VARNAME(broadphase, "BroadPhase");
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
    if (narrowPhaseDetection == nullptr)
    {
        return; // can't go further
    }

    msg_info_when(d_doPrintInfoMessage.getValue())
        << "doCollisionDetection, NarrowPhaseDetection "<<narrowPhaseDetection->getName();

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("NarrowPhase");
#endif
    {
        SCOPED_TIMER_VARNAME(narrowphase, "NarrowPhase");
        intersectionMethod->beginNarrowPhase();
        narrowPhaseDetection->beginNarrowPhase();
        const type::vector<std::pair<CollisionModel*, CollisionModel*> >& vectCMPair = broadPhaseDetection->getCollisionModelPairs();

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

void CollisionPipeline::doCollisionResponse()
{
    core::objectmodel::BaseContext* scene = getContext();
    // then we start the creation of contacts
    if (narrowPhaseDetection == nullptr || contactManager == nullptr)
    {
        return; // can't go further
    }

    msg_info_when(d_doPrintInfoMessage.getValue())
        << "Create Contacts " << contactManager->getName() ;

    {
        SCOPED_TIMER_VARNAME(createContactsTimer, "CreateContacts");
        contactManager->createContacts(narrowPhaseDetection->getDetectionOutputs());
    }

    // finally we start the creation of collisionGroup

    const type::vector<Contact::SPtr>& contacts = contactManager->getContacts();

    // First we remove all contacts with non-simulated objects and directly add them
    type::vector<Contact::SPtr> notStaticContacts;

    {
        SCOPED_TIMER_VARNAME(createStaticObjectsResponseTimer, "CreateStaticObjectsResponse");
        for (const auto& contact : contacts)
        {
            const auto collisionModels = contact->getCollisionModels();
            if (collisionModels.first != nullptr && !collisionModels.first->isSimulated())
            {
                contact->createResponse(collisionModels.second->getContext());
            }
            else if (collisionModels.second != nullptr && !collisionModels.second->isSimulated())
            {
                contact->createResponse(collisionModels.first->getContext());
            }
            else
            {
                notStaticContacts.push_back(contact);
            }
        }
    }

    if (groupManager == nullptr)
    {
        SCOPED_TIMER_VARNAME(createResponseTimer, "CreateMovingObjectsResponse");

        msg_info_when(d_doPrintInfoMessage.getValue())
            << "Linking all contacts to Scene" ;

        for (const auto& contact : notStaticContacts)
        {
            contact->createResponse(scene);
        }
    }
    else
    {
        msg_info_when(d_doPrintInfoMessage.getValue())
           << "Create Groups "<<groupManager->getName();

        groupManager->createGroups(scene, notStaticContacts);
    }
}

std::set< std::string > CollisionPipeline::getResponseList() const
{
    std::set< std::string > listResponse;
    core::collision::Contact::Factory::iterator it;
    for (it=core::collision::Contact::Factory::getInstance()->begin(); it!=core::collision::Contact::Factory::getInstance()->end(); ++it)
    {
        listResponse.insert(it->first);
    }
    return listResponse;
}

} // namespace sofa::component::collision::detection::algorithm
