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
#include <sofa/component/collision/response/contact/ContactListener.h>

#include <sofa/core/collision/NarrowPhaseDetection.h>

#include <sofa/core/CollisionModel.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/CollisionBeginEvent.h>
#include <sofa/simulation/CollisionEndEvent.h>
#include <sofa/type/Vec.h>

#include <tuple>

namespace sofa::component::collision::response::contact
{
using namespace sofa::core;
using namespace sofa::core::collision;

int ContactListenerClass = core::RegisterObject("ContactListener .. ").add< ContactListener >();

ContactListener::ContactListener()
    :  l_collisionModel1(initLink("collisionModel1", "Collision model one"))
    ,  l_collisionModel2(initLink("collisionModel2", "Collision model two"))
    ,  l_narrowPhase(initLink("narrowPhase", "Use a narrow phase (default=search in context)"))
{
}

ContactListener::~ContactListener()
{
}

void ContactListener::init(void)
{
    d_componentState = sofa::core::objectmodel::ComponentState::Valid;
    Inherit1::init();

    if(!l_collisionModel1)
        l_collisionModel1 = getContext()->get<core::CollisionModel>();

    if(!l_collisionModel1)
    {
        msg_error() << "missing 'collisionModel1' parameter and nothing in the context";
        d_componentState = sofa::core::objectmodel::ComponentState::Invalid;
        return;
    }

    if(!l_collisionModel2)
        l_collisionModel2 = getContext()->get<core::CollisionModel>();

    if(!l_collisionModel2)
    {
        msg_error() << "missing 'collisionModel2' parameter and nothing in the context";
        d_componentState = sofa::core::objectmodel::ComponentState::Invalid;
        return;
    }

    if( !l_narrowPhase )
        l_narrowPhase = getContext()->get<core::collision::NarrowPhaseDetection>();

    if ( l_narrowPhase )
    {
        // add to the event listening
        f_listening.setValue(true);
    }
}

void ContactListener::handleEvent( core::objectmodel::Event* _event )
{
    if(!isComponentStateValid())
        return;

    if (simulation::CollisionBeginEvent::checkEventType(_event))
    {
        m_ContactsVector.swap(m_ContactsVectorBuffer);
        m_ContactsVector.clear();
    }

    else if (simulation::CollisionEndEvent::checkEventType(_event))
    {

        const auto& detectionOutputsMap = l_narrowPhase->getDetectionOutputs();

        if ( detectionOutputsMap.empty() )
        {
            endContact(nullptr);
            return;
        }

        if  ( l_collisionModel2 == nullptr )
        {
            //// check only one collision model
            for (const auto & it : detectionOutputsMap)
            {
                const CollisionModel* collMod1 = it.first.first;
                const CollisionModel* collMod2 = it.first.second;

                if ( l_collisionModel1 == collMod1 || l_collisionModel1 == collMod2 )
                {
                    if ( const type::vector<DetectionOutput>* contacts = dynamic_cast<type::vector<DetectionOutput>*>(it.second) )
                    {
                        m_ContactsVector.push_back( *contacts );
                    }
                }
            }
        }
        else
        {
            // check both collision models
            for (const auto & it : detectionOutputsMap)
            {
                const CollisionModel* collMod1 = it.first.first;
                const CollisionModel* collMod2 = it.first.second;

                if ( (l_collisionModel1==collMod1 && l_collisionModel2==collMod2) || (l_collisionModel1==collMod2 && l_collisionModel2==collMod1) )
                {
                    if ( const type::vector<DetectionOutput>* contacts = dynamic_cast<type::vector<DetectionOutput>*>(it.second) )
                    {
                        m_ContactsVector.push_back( *contacts );
                    }
                }
            }
        }
        beginContact(m_ContactsVector);
    }
}

sofa::Size ContactListener::getNumberOfContacts() const
{
    if (!m_ContactsVectorBuffer.empty())
    {
        const sofa::Size numberOfContacts = m_ContactsVectorBuffer[0].size();
        if (0 < numberOfContacts && ((numberOfContacts <= l_collisionModel1->getSize()) || (numberOfContacts <= l_collisionModel2->getSize()))){
            return numberOfContacts;
        }
        else {
            return 0;
        }
    }
    else {
        return 0;
    }
}

type::vector<double> ContactListener::getDistances() const
{
    type::vector<double> distances;
    const sofa::Size numberOfContacts = getNumberOfContacts();
    if (0 < numberOfContacts){ // can be 0
        distances.reserve(numberOfContacts);
        for (const auto& c: m_ContactsVectorBuffer[0]){
            distances.emplace_back(c.value);
        }
    }
    return distances;
}

std::vector<std::tuple<unsigned int, sofa::type::Vec3, unsigned int, sofa::type::Vec3>> ContactListener::getContactPoints() const
{
    std::vector<std::tuple<unsigned int, sofa::type::Vec3, unsigned int, sofa::type::Vec3>> contactPoints;
    const sofa::Size numberOfContacts = getNumberOfContacts();
    if (0 < numberOfContacts){ // can be 0
        contactPoints.reserve(numberOfContacts);
        for (const auto& c: m_ContactsVectorBuffer[0]){
            unsigned int firstID = l_collisionModel1 != c.elem.first.getCollisionModel();
            unsigned int secondID = !firstID;
            contactPoints.emplace_back(firstID, c.point[0], secondID, c.point[1]);
        }
    }
    return contactPoints;
}

std::vector<std::tuple<unsigned int, unsigned int, unsigned int, unsigned int>> ContactListener::getContactElements() const
{
    std::vector<std::tuple<unsigned int, unsigned int, unsigned int, unsigned int>> contactElements;
    const sofa::Size numberOfContacts = getNumberOfContacts();
    if (0 < numberOfContacts){ // can be 0
        contactElements.reserve(numberOfContacts);
        for (const auto& c: m_ContactsVectorBuffer[0]){
            unsigned int firstID = l_collisionModel1 != c.elem.first.getCollisionModel();
            unsigned int secondID = !firstID;
            contactElements.emplace_back(firstID, c.elem.first.getIndex(), secondID, c.elem.second.getIndex());
        }
    }
    return contactElements;
}

type::vector<type::vector<DetectionOutput>> ContactListener::getContactsVector() const
{
    return m_ContactsVector;
}

} // namespace sofa::component::collision::response::contact
