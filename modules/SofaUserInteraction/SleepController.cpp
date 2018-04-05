/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaUserInteraction/SleepController.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/collision/Contact.h>
#include <sofa/core/collision/ContactManager.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/simulation/CollisionEndEvent.h>

namespace sofa
{

namespace component
{

namespace controller
{

template <class DataTypes>
StateTester<DataTypes>::~StateTester()
{
}

template <class DataTypes>
bool StateTester<DataTypes>::canConvert(core::behavior::BaseMechanicalState* baseState)
{
    return dynamic_cast< core::behavior::MechanicalState<DataTypes>* >(baseState) != NULL;
}

template <class DataTypes>
bool StateTester<DataTypes>::wantsToSleep(core::behavior::BaseMechanicalState* baseState, SReal speedThreshold, SReal /*rotationThreshold*/)
{
    core::behavior::MechanicalState<DataTypes>* state = dynamic_cast< core::behavior::MechanicalState<DataTypes>* >(baseState);
    if (!state)
        return false;

    typedef typename DataTypes::VecDeriv VecDeriv;

    const VecDeriv& vecVel = state->read(core::ConstVecDerivId::velocity())->getValue();
    SReal maxVelocity = 0;
    for (unsigned int i = 0, nb = vecVel.size(); i < nb; ++i)
        maxVelocity = std::max(maxVelocity, vecVel[i].norm2());

    return maxVelocity < (speedThreshold * speedThreshold);
}

template <>
bool StateTester<defaulttype::Rigid2Types>::wantsToSleep(core::behavior::BaseMechanicalState* baseState, SReal speedThreshold, SReal rotationThreshold)
{
    typedef defaulttype::Rigid2Types DataTypes;
    core::behavior::MechanicalState<DataTypes>* state = dynamic_cast< core::behavior::MechanicalState<DataTypes>* >(baseState);
    if (!state)
        return false;

    const DataTypes::VecDeriv& vecVel = state->read(core::ConstVecDerivId::velocity())->getValue();
    SReal maxSpeed = 0, maxRotation = 0;
    for (unsigned int i = 0, nb = vecVel.size(); i < nb; ++i)
    {
        SReal speed = vecVel[i].getVCenter().norm2();
        maxSpeed = std::max(maxSpeed, speed);

        SReal rotation = vecVel[i].getVOrientation(); // not squared, it is directly a scalar
        maxRotation = std::max(maxRotation, rotation);
    }

    return maxSpeed < (speedThreshold * speedThreshold)
        && (!rotationThreshold || maxRotation < rotationThreshold);
}

template <>
bool StateTester<defaulttype::Rigid3Types>::wantsToSleep(core::behavior::BaseMechanicalState* baseState, SReal speedThreshold, SReal rotationThreshold)
{
    typedef defaulttype::Rigid3Types DataTypes;
    core::behavior::MechanicalState<DataTypes>* state = dynamic_cast< core::behavior::MechanicalState<DataTypes>* >(baseState);
    if (!state)
        return false;

    const DataTypes::VecDeriv& vecVel = state->read(core::ConstVecDerivId::velocity())->getValue();
    SReal maxSpeed = 0, maxRotation = 0;
    for (unsigned int i = 0, nb = vecVel.size(); i < nb; ++i)
    {
        SReal speed = vecVel[i].getVCenter().norm2();
        maxSpeed = std::max(maxSpeed, speed);

        SReal rotation = vecVel[i].getVOrientation().norm2();
        maxRotation = std::max(maxRotation, rotation);
    }

    return maxSpeed < (speedThreshold * speedThreshold)
        && (!rotationThreshold || maxRotation < (rotationThreshold * rotationThreshold));
}

SleepController::SleepController()
    : d_minTimeSinceWakeUp(initData(&d_minTimeSinceWakeUp, 0.1, "minTimeSinceWakeUp", "Do not do anything before objects have been moving for this duration"))
    , d_speedThreshold(initData(&d_speedThreshold, (SReal)0.001, "immobileThreshold", "Speed value under which we consider a particule to be immobile"))
    , d_rotationThreshold(initData(&d_rotationThreshold, (SReal)0.0, "rotationThreshold", "If non null, this is the rotation speed value under which we consider a particule to be immobile"))
{
    f_listening.setValue(true);

    // These are the supported types (add others if necessary, just be sure StateTester::wantsToSleep is correctly implemented)
    addType<defaulttype::Vec1Types>();
    addType<defaulttype::Vec2Types>();
    addType<defaulttype::Vec3Types>();
    addType<defaulttype::Vec6Types>();
    addType<defaulttype::Rigid2Types>();
    addType<defaulttype::Rigid3Types>();
}

SleepController::~SleepController()
{

}

void SleepController::init()
{
    m_statesThatCanSleep.clear();
    m_contextsThatCanSleep.clear();
    m_correspondingTesters.clear();

    // Find all nodes with the flag "canChangeSleepingState"
    StatesThatCanSleep tempStates;
    GetStatesThatCanSleep(core::ExecParams::defaultInstance(), tempStates).execute(getContext()->getRootContext());

    // Find the corresponding template in each mechanical state we are monitoring
    for (unsigned int i = 0, nbStates = tempStates.size(); i < nbStates; ++i)
    {
        core::behavior::BaseMechanicalState* state = tempStates[i];
        bool found = false;
        for (unsigned int j = 0, nbTesters = m_stateTesters.size(); j < nbTesters; ++j)
        {
            StateTesterPtr tester = m_stateTesters[j];
            if (tester->canConvert(state))
            {
                found = true;
                m_statesThatCanSleep.push_back(state);
                m_contextsThatCanSleep.push_back(state->getContext());
                m_correspondingTesters.push_back(tester);
                break;
            }
        }

        if (!found)
            serr << "SleepController can not control node " << state->getContext()->getName() << " of type " << state->getClass()->templateName << sendl;
    }

    m_timeSinceWakeUp.assign(m_contextsThatCanSleep.size(), 0.0);

    // If it is the first time doing the init, copy the initial state values
    if (m_initialState.empty())
    {
        for (unsigned int i = 0, nbContexts = m_contextsThatCanSleep.size(); i < nbContexts; ++i)
            m_initialState.push_back(m_contextsThatCanSleep[i]->isSleeping());
    }

    sout << "found " << m_statesThatCanSleep.size() << " nodes that can change their sleep state" << sendl;
}

void SleepController::reset()
{
    // Reset the states of the nodes to the initial values
    if (m_initialState.size() == m_contextsThatCanSleep.size())
    {
        for (unsigned int i = 0, nbContexts = m_contextsThatCanSleep.size(); i < nbContexts; ++i)
            m_contextsThatCanSleep[i]->setSleeping(m_initialState[i]);
    }

    // Reset time since wake up to 0
    for (unsigned int i = 0, nb = m_timeSinceWakeUp.size(); i < nb; ++i)
            m_timeSinceWakeUp[i] = 0.0;
}

void SleepController::handleEvent(core::objectmodel::Event* event)
{
    if (sofa::simulation::AnimateBeginEvent::checkEventType(event))
        putNodesToSleep();
    else if (sofa::simulation::CollisionEndEvent::checkEventType(event))
    {
        wakeUpNodes();
        updateSleepStatesRecursive();
    }
    else if (sofa::simulation::AnimateEndEvent::checkEventType(event))
        updateTimeSinceWakeUp();
}

void SleepController::putNodesToSleep()
{
    // Don't do anything before the objects have the time to be moving
    double minTimeSinceWakeUp = d_minTimeSinceWakeUp.getValue();

    SReal speedThreshold = d_speedThreshold.getValue();
    SReal rotThreshold = d_rotationThreshold.getValue();
    for (unsigned int i = 0, nbStates = m_statesThatCanSleep.size(); i < nbStates; ++i)
    {
        core::behavior::BaseMechanicalState* state = m_statesThatCanSleep[i];
        core::objectmodel::BaseContext* context = m_contextsThatCanSleep[i];

        if (!context->isSleeping() &&
            m_timeSinceWakeUp[i] >= minTimeSinceWakeUp &&
            m_correspondingTesters[i]->wantsToSleep(state, speedThreshold, rotThreshold))
        {
            context->setSleeping(true);

            dmsg_info() << " put node " << context->getName() << " to sleep" ;
        }
    }
}

void SleepController::wakeUpNodes()
{
    std::vector<BaseContexts> wakeupPairs; // For each context that can sleep, all linked contexts
    wakeupPairs.resize(m_statesThatCanSleep.size());

    collectWakeupPairs(wakeupPairs);

    //TODO(dmarchal) remove it or uncomment it !
    // Debug of the pairs
    /*	if (notMuted())
    {
        for (unsigned int i = 0, nbWakeupPairs = wakeupPairs.size(); i < nbWakeupPairs; ++i)
        {
            const BaseContexts& wakeupPairRef = wakeupPairs[i];
            if (!wakeupPairRef.empty())
            {
                std::stringstream tmp
                for (unsigned int j = 0, nbLinks = wakeupPairRef.size(); j < nbLinks; ++j)
                    tmp << m_contextsThatCanSleep[i]->getName() << " --> " << wakeupPairRef[j]->getName() << msgendl;
                msg_info() << tmp ;
            }
        }
    } */

    // We use an iterative process to wake up nodes until we don't change any state
    bool changed = false;
    do
    {
        changed = false;
        unsigned int nbContexts = m_contextsThatCanSleep.size();
        for (unsigned int i = 0; i < nbContexts; ++i)
        {
            core::objectmodel::BaseContext* context = m_contextsThatCanSleep[i];
            if (!context->isSleeping())
                continue;

            const BaseContexts& wakeupPair = wakeupPairs[i];
            for (unsigned int j = 0, nbLinks = wakeupPair.size(); j < nbLinks; ++j)
            {
                if (!wakeupPair[j]->isSleeping())
                {
                    changed = true;
                    context->setSleeping(false);

                    msg_info() << "Wake up node " << context->getName() ;

                    break;
                }
            }
        }
    } while (changed);
}

void SleepController::updateTimeSinceWakeUp()
{
    for (unsigned int i = 0, nbContexts = m_contextsThatCanSleep.size(); i < nbContexts; ++i)
    {
        core::objectmodel::BaseContext* context = m_contextsThatCanSleep[i];
        if (context->isSleeping())
            m_timeSinceWakeUp[i] = 0.0;
        else
            m_timeSinceWakeUp[i] += context->getDt();
    }
}

void SleepController::updateSleepStatesRecursive()
{
    UpdateAllSleepStates(core::ExecParams::defaultInstance()).execute(getContext()->getRootContext());
}

void SleepController::collectWakeupPairs(std::vector<BaseContexts>& wakeupPairs)
{
    std::vector<core::collision::Contact::SPtr> contacts;
    core::collision::ContactManager* contactManager;
    core::objectmodel::BaseContext* root = getContext()->getRootContext();

    root->get(contactManager);
    if (contactManager)
        contacts = contactManager->getContacts();
    else
        root->get<core::collision::Contact>(&contacts, core::objectmodel::BaseContext::SearchDown);

    for (unsigned int i = 0, nbContacts = contacts.size(); i < nbContacts; ++i)
    {
        std::pair<core::CollisionModel*, core::CollisionModel*> collisionModels = contacts[i]->getCollisionModels();

        addWakeupPair(wakeupPairs, collisionModels.first->getContext(), collisionModels.first->isMoving(),
                                    collisionModels.second->getContext(), collisionModels.second->isMoving());
    }
}

void SleepController::addWakeupPair(std::vector<BaseContexts>& wakeupPairs, core::objectmodel::BaseContext* context1, bool moving1, core::objectmodel::BaseContext* context2, bool moving2)
{
    // NB: an unmoving object never wakeups a moving one, but you can (and should) allow unmoving objects to sleep,
    // and thus it must be possible to wake them up when entering contact with a moving object.

    if (!moving1 && !moving2)
        return;

    context1 = getParentContextThatCanSleep(context1);
    context2 = getParentContextThatCanSleep(context2);
    if (context1 == NULL || context2 == NULL)
        return;

    BaseContexts::const_iterator contextsBegin = m_contextsThatCanSleep.begin();
    BaseContexts::const_iterator contextsEnd = m_contextsThatCanSleep.end();

    if (moving2)
    {
        BaseContexts::const_iterator iter = std::find(contextsBegin, contextsEnd, context1);
        if (iter != contextsEnd)
        {
            int index = iter - contextsBegin;
            BaseContexts& wakeupPairRef = wakeupPairs[index];
            if(std::find(wakeupPairRef.begin(), wakeupPairRef.end(), context2) == wakeupPairRef.end()) // No duplicates
                wakeupPairRef.push_back(context2);
        }
    }

    if (moving1)
    {
        BaseContexts::const_iterator iter = std::find(contextsBegin, contextsEnd, context2);
        if (iter != contextsEnd)
        {
            int index = iter - contextsBegin;
            BaseContexts& wakeupPairRef = wakeupPairs[index];
            if(std::find(wakeupPairRef.begin(), wakeupPairRef.end(), context1) == wakeupPairRef.end()) // No duplicates
                wakeupPairRef.push_back(context1);
        }
    }
}

core::objectmodel::BaseContext* SleepController::getParentContextThatCanSleep(core::objectmodel::BaseContext* context)
{
    // Start the search from the given node
    core::objectmodel::BaseNode* node =context->toBaseNode();
    if (!node)
        return context;

    std::deque<core::objectmodel::BaseNode*> parents;
    parents.push_back(node);

    // We go up one level at a time
    while (!parents.empty())
    {
        core::objectmodel::BaseNode* currentNode = parents.front();
        parents.pop_front();

        // Test all contexts that can sleep
        core::objectmodel::BaseContext* context = currentNode->getContext();
        if (context->canChangeSleepingState())
            return context;

        // Add all direct parents
        core::objectmodel::BaseNode::Parents currentParents = currentNode->getParents();
        std::copy(currentParents.begin(), currentParents.end(), std::back_inserter(parents));
    }

    return context; // Not found
}

GetStatesThatCanSleep::GetStatesThatCanSleep(const core::ExecParams* params, std::vector<core::behavior::BaseMechanicalState*>& states)
    : simulation::Visitor(params)
    , m_states(states)
{}

void GetStatesThatCanSleep::processNodeBottomUp(simulation::Node* node)
{
    if (node->canChangeSleepingState() && node->mechanicalState != NULL)
        m_states.push_back(node->mechanicalState.get());
}

UpdateAllSleepStates::UpdateAllSleepStates(const core::ExecParams* params)
    : simulation::Visitor(params)
{}

simulation::Visitor::Result UpdateAllSleepStates::processNodeTopDown(simulation::Node* node)
{
    if (!node->canChangeSleepingState()) // nodes that can change their sleep state are directly manipulated and do not depend on their parents
    {
        bool sleeping = false;

        core::objectmodel::BaseNode::Parents parents = node->getParents();
        if (parents.size())
        {
            sleeping = true;
            for ( unsigned int i = 0; i < parents.size(); i++ )
            {
                sleeping &= parents[i]->getContext()->isSleeping();
            }
        }

        node->getContext()->setSleeping(sleeping);
    }

    return RESULT_CONTINUE;
}

int SleepControllerClass = core::RegisterObject("A controller that puts node into sleep when the objects are not moving, and wake them up again when there are in collision with a moving object")
.add< SleepController >();

SOFA_DECL_CLASS(SleepController)

} // namespace controller

} // namepace component

} // namespace sofa
