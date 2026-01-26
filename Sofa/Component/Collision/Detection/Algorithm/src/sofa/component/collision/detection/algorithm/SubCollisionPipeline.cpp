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
#include <sofa/component/collision/detection/algorithm/SubCollisionPipeline.h>

#include <sofa/core/ObjectFactory.h>

#include <sofa/core/collision/Contact.h>

#include <sofa/simulation/Node.h>

#include <sofa/helper/ScopedAdvancedTimer.h>
using sofa::helper::ScopedAdvancedTimer ;

#include <sofa/helper/AdvancedTimer.h>


namespace sofa::component::collision::detection::algorithm
{

using namespace sofa;
using namespace sofa::core;
using namespace sofa::core::collision;

void registerSubCollisionPipeline(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Collision pipeline to be used with CompositeCollisionPipeline.")
        .add< SubCollisionPipeline >());
}

SubCollisionPipeline::SubCollisionPipeline()
    : Inherited()
    , d_depth(initData(&d_depth, s_defaultDepthValue, "depth", +("Max depth of bounding trees. (default=" + std::to_string(s_defaultDepthValue) + ", min=?, max=?)").c_str()))
    , l_collisionModels(initLink("collisionModels", "List of collision models to consider in this pipeline"))
    , l_intersectionMethod(initLink("intersectionMethod", "Intersection method to use in this pipeline"))
    , l_contactManager(initLink("contactManager", "Contact manager to use in this pipeline"))
    , l_broadPhaseDetection(initLink("broadPhaseDetection", "Broad phase detection to use in this pipeline"))
    , l_narrowPhaseDetection(initLink("narrowPhaseDetection", "Narrow phase detection to use in this pipeline"))
{
}

void SubCollisionPipeline::doInit()
{
    bool validity = true;

    //Check given parameters
    if (l_collisionModels.size() == 0)
    {
        msg_warning() << "At least one CollisionModel is required to compute collision detection.";
        validity = false;
    }
    
    if (!l_intersectionMethod)
    {
        msg_warning() << "An Intersection detection component is required to compute collision detection.";
        validity = false;
    }
    
    if (!l_contactManager)
    {
        msg_warning() << "A contact manager component is required to compute collision detection.";
        validity = false;
    }
    
    if (!l_broadPhaseDetection)
    {
        msg_warning() << "A BroadPhase component is required to compute collision detection.";
        validity = false;
    }
    if (!l_narrowPhaseDetection)
    {
        msg_warning() << "A NarrowPhase component is required to compute collision detection.";
        validity = false;
    }

    if (!validity)
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
    }
    else
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
    }

}

void SubCollisionPipeline::computeCollisionReset()
{
    if (!this->isComponentStateValid())
        return;

    msg_info() << "SubCollisionPipeline::doCollisionReset";

    l_broadPhaseDetection->setIntersectionMethod(l_intersectionMethod.get());
    l_narrowPhaseDetection->setIntersectionMethod(l_intersectionMethod.get());
    l_contactManager->setIntersectionMethod(l_intersectionMethod.get());

    // clear all contacts
    const type::vector<Contact::SPtr>& contacts = l_contactManager->getContacts();
    for (const auto& contact : contacts)
    {
        if (contact != nullptr)
        {
            contact->removeResponse();
        }
    }
}

void SubCollisionPipeline::computeCollisionDetection()
{
    SCOPED_TIMER_VARNAME(docollisiontimer, "doCollisionDetection");

    if (!this->isComponentStateValid())
        return;

    msg_info()  << "doCollisionDetection, compute Bounding Trees" ;

    // First, we compute a bounding volume for the collision model (for example bounding sphere)
    // or we have loaded a collision model that knows its other model
    
    type::vector<CollisionModel*> vectBoundingVolume;
    {
        SCOPED_TIMER_VARNAME(bboxtimer, "ComputeBoundingTree");

        const bool continuous = l_intersectionMethod->useContinuous();
        const auto continuousIntersectionType = l_intersectionMethod->continuousIntersectionType();
        const SReal dt       = getContext()->getDt();

        int nActive = 0;

        const int used_depth = (
                    (l_broadPhaseDetection->needsDeepBoundingTree()) ||
                    (l_narrowPhaseDetection->needsDeepBoundingTree())
            ) ? d_depth.getValue() : 0;

        for (auto it = l_collisionModels.begin(); it != l_collisionModels.end(); ++it)
        {
            msg_info() << "doCollisionDetection, consider model" ;

            if (!(*it)->isActive()) continue;

            if (continuous)
            {
                const std::string msg = "Compute Continuous BoundingTree: " + (*it)->getName();
                ScopedAdvancedTimer continuousBoundingTreeTimer(msg.c_str());
                (*it)->computeContinuousBoundingTree(dt, continuousIntersectionType, used_depth);
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


        msg_info() << "doCollisionDetection, Computed "<<nActive<<" Bounding Boxes." ;
    }

    msg_info()  << "doCollisionDetection, BroadPhaseDetection "<<l_broadPhaseDetection->getName();

    {
        SCOPED_TIMER_VARNAME(broadphase, "BroadPhase");
        l_intersectionMethod->beginBroadPhase();
        l_broadPhaseDetection->beginBroadPhase();
        l_broadPhaseDetection->addCollisionModels(vectBoundingVolume);  // detection is done there
        l_broadPhaseDetection->endBroadPhase();
        l_intersectionMethod->endBroadPhase();
    }

    msg_info() << "doCollisionDetection, NarrowPhaseDetection "<< l_narrowPhaseDetection->getName();

    {
        SCOPED_TIMER_VARNAME(narrowphase, "NarrowPhase");
        l_intersectionMethod->beginNarrowPhase();
        l_narrowPhaseDetection->beginNarrowPhase();
        const type::vector<std::pair<CollisionModel*, CollisionModel*> >& vectCMPair = l_broadPhaseDetection->getCollisionModelPairs();

        msg_info()  << "doCollisionDetection, "<< vectCMPair.size()<<" colliding model pairs" ;

        l_narrowPhaseDetection->addCollisionPairs(vectCMPair);
        l_narrowPhaseDetection->endNarrowPhase();
        l_intersectionMethod->endNarrowPhase();
    }

}

void SubCollisionPipeline::computeCollisionResponse()
{
    if (!this->isComponentStateValid())
        return;

    core::objectmodel::BaseContext* scene = getContext();

    msg_info() << "Create Contacts " << l_contactManager->getName() ;

    {
        SCOPED_TIMER_VARNAME(createContactsTimer, "CreateContacts");
        l_contactManager->createContacts(l_narrowPhaseDetection->getDetectionOutputs());
    }

    // finally we start the creation of collisionGroup
    const type::vector<Contact::SPtr>& contacts = l_contactManager->getContacts();

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

    SCOPED_TIMER_VARNAME(createResponseTimer, "CreateMovingObjectsResponse");

    msg_info() << "Linking all contacts to Scene" ;

    for (const auto& contact : notStaticContacts)
    {
        contact->createResponse(scene);
    }
}


std::vector<sofa::core::CollisionModel*> SubCollisionPipeline::getCollisionModels()
{
    std::vector<sofa::core::CollisionModel*> collisionModels;
    collisionModels.reserve(l_collisionModels.getSize());
    for(auto* collisionModel : l_collisionModels)
    {
        collisionModels.push_back(collisionModel);
    }
    return collisionModels;
}

} // namespace sofa::component::collision::detection::algorithm
