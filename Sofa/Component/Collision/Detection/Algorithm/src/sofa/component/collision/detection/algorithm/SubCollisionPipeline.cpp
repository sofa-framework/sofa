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
    , d_depth(initData(&d_depth, s_defaultDepthValue, "depth", ("Max depth of bounding trees. (default=" + std::to_string(s_defaultDepthValue) + ", min=?, max=?)").c_str()))
    , l_collisionModels(initLink("collisionModels", "List of collision models to consider in this pipeline"))
    , l_intersectionMethod(initLink("intersectionMethod", "Intersection method to use in this pipeline"))
    , l_contactManager(initLink("contactManager", "Contact manager to use in this pipeline"))
    , l_broadPhaseDetection(initLink("broadPhaseDetection", "Broad phase detection to use in this pipeline"))
    , l_narrowPhaseDetection(initLink("narrowPhaseDetection", "Narrow phase detection to use in this pipeline"))
{
}

/**
 * @brief Validates that all required components are properly linked.
 *
 * Checks for the presence of all mandatory links:
 * - At least one collision model
 * - An intersection method
 * - A contact manager
 * - A broad phase detection component
 * - A narrow phase detection component
 *
 * Sets the component state to Invalid if any required component is missing,
 * which prevents the pipeline from executing collision detection.
 */
void SubCollisionPipeline::doInit()
{
    bool validity = true;

    // Validate all required links are set
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

    // Set component state based on validation results
    if (!validity)
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
    }
    else
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
    }

}

/**
 * @brief Resets the collision state by clearing all existing contact responses.
 *
 * This method prepares for a new collision detection cycle by:
 * 1. Propagating the intersection method to all detection components
 * 2. Removing all contact responses created during the previous time step
 *
 * This ensures a clean slate before new collisions are detected.
 */
void SubCollisionPipeline::computeCollisionReset()
{
    if (!this->isComponentStateValid())
        return;

    msg_info() << "SubCollisionPipeline::doCollisionReset";

    // Propagate the intersection method to all collision detection components
    l_broadPhaseDetection->setIntersectionMethod(l_intersectionMethod.get());
    l_narrowPhaseDetection->setIntersectionMethod(l_intersectionMethod.get());
    l_contactManager->setIntersectionMethod(l_intersectionMethod.get());

    // Remove all contact responses from the previous time step
    const type::vector<Contact::SPtr>& contacts = l_contactManager->getContacts();
    for (const auto& contact : contacts)
    {
        if (contact != nullptr)
        {
            contact->removeResponse();
        }
    }
}

/**
 * @brief Performs collision detection in two phases: broad phase and narrow phase.
 */
void SubCollisionPipeline::computeCollisionDetection()
{
    SCOPED_TIMER_VARNAME(docollisiontimer, "doCollisionDetection");

    if (!this->isComponentStateValid())
        return;

    msg_info()  << "doCollisionDetection, compute Bounding Trees" ;

    // Phase 1: Compute bounding volumes for all collision models
    // These hierarchical structures enable efficient spatial queries
    type::vector<CollisionModel*> vectBoundingVolume;
    {
        SCOPED_TIMER_VARNAME(bboxtimer, "ComputeBoundingTree");

        // Check if continuous collision detection (CCD) is enabled
        const bool continuous = l_intersectionMethod->useContinuous();
        const auto continuousIntersectionType = l_intersectionMethod->continuousIntersectionType();
        const SReal dt       = getContext()->getDt();

        int nActive = 0;

        // Use full tree depth only if detection algorithms require it, otherwise use depth 0
        const int used_depth = (
                    (l_broadPhaseDetection->needsDeepBoundingTree()) ||
                    (l_narrowPhaseDetection->needsDeepBoundingTree())
            ) ? d_depth.getValue() : 0;

        // Iterate through all linked collision models
        for (auto it = l_collisionModels.begin(); it != l_collisionModels.end(); ++it)
        {
            msg_info() << "doCollisionDetection, consider model" ;

            // Skip inactive models
            if (!(*it)->isActive()) continue;

            if (continuous)
            {
                // CCD: Compute swept bounding volumes that cover the motion trajectory
                const std::string msg = "Compute Continuous BoundingTree: " + (*it)->getName();
                ScopedAdvancedTimer continuousBoundingTreeTimer(msg.c_str());
                (*it)->computeContinuousBoundingTree(dt, continuousIntersectionType, used_depth);
            }
            else
            {
                // Discrete: Compute bounding volumes at current positions
                std::string msg = "Compute BoundingTree: " + (*it)->getName();
                ScopedAdvancedTimer boundingTreeTimer(msg.c_str());
                (*it)->computeBoundingTree(used_depth);
            }

            // getFirst() returns the root of the bounding tree hierarchy
            vectBoundingVolume.push_back ((*it)->getFirst());
            ++nActive;
        }


        msg_info() << "doCollisionDetection, Computed "<<nActive<<" Bounding Boxes." ;
    }

    // Phase 2: Broad Phase Detection
    // Quickly finds pairs of models whose bounding volumes overlap
    msg_info()  << "doCollisionDetection, BroadPhaseDetection "<<l_broadPhaseDetection->getName();

    {
        SCOPED_TIMER_VARNAME(broadphase, "BroadPhase");
        l_intersectionMethod->beginBroadPhase();
        l_broadPhaseDetection->beginBroadPhase();
        l_broadPhaseDetection->addCollisionModels(vectBoundingVolume);  // Actual detection happens here
        l_broadPhaseDetection->endBroadPhase();
        l_intersectionMethod->endBroadPhase();
    }

    // Phase 3: Narrow Phase Detection
    // Performs precise intersection tests on potentially colliding pairs
    msg_info() << "doCollisionDetection, NarrowPhaseDetection "<< l_narrowPhaseDetection->getName();

    {
        SCOPED_TIMER_VARNAME(narrowphase, "NarrowPhase");
        l_intersectionMethod->beginNarrowPhase();
        l_narrowPhaseDetection->beginNarrowPhase();

        // Get the pairs identified by broad phase
        const type::vector<std::pair<CollisionModel*, CollisionModel*> >& vectCMPair = l_broadPhaseDetection->getCollisionModelPairs();

        msg_info()  << "doCollisionDetection, "<< vectCMPair.size()<<" colliding model pairs" ;

        // Perform precise intersection tests on each pair
        l_narrowPhaseDetection->addCollisionPairs(vectCMPair);
        l_narrowPhaseDetection->endNarrowPhase();
        l_intersectionMethod->endNarrowPhase();
    }

}

/**
 * @brief Creates collision responses based on detected contacts.
 */
void SubCollisionPipeline::computeCollisionResponse()
{
    if (!this->isComponentStateValid())
        return;

    core::objectmodel::BaseContext* scene = getContext();

    msg_info() << "Create Contacts " << l_contactManager->getName() ;

    // Create contact objects from narrow phase detection results
    {
        SCOPED_TIMER_VARNAME(createContactsTimer, "CreateContacts");
        l_contactManager->createContacts(l_narrowPhaseDetection->getDetectionOutputs());
    }

    const type::vector<Contact::SPtr>& contacts = l_contactManager->getContacts();

    // Separate contacts into two categories based on whether they involve static objects
    type::vector<Contact::SPtr> notStaticContacts;

    // Process contacts involving static (non-simulated) objects first
    // These get their response attached to the simulated object's context
    {
        SCOPED_TIMER_VARNAME(createStaticObjectsResponseTimer, "CreateStaticObjectsResponse");
        for (const auto& contact : contacts)
        {
            const auto collisionModels = contact->getCollisionModels();
            if (collisionModels.first != nullptr && !collisionModels.first->isSimulated())
            {
                // First model is static, attach response to second model's context
                contact->createResponse(collisionModels.second->getContext());
            }
            else if (collisionModels.second != nullptr && !collisionModels.second->isSimulated())
            {
                // Second model is static, attach response to first model's context
                contact->createResponse(collisionModels.first->getContext());
            }
            else
            {
                // Both models are simulated, handle separately
                notStaticContacts.push_back(contact);
            }
        }
    }

    // Process contacts between two simulated (moving) objects
    // These get their response attached to the scene context
    SCOPED_TIMER_VARNAME(createResponseTimer, "CreateMovingObjectsResponse");

    msg_info() << "Linking all contacts to Scene" ;

    for (const auto& contact : notStaticContacts)
    {
        contact->createResponse(scene);
    }
}


/// Returns the list of collision models explicitly linked to this pipeline.
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
