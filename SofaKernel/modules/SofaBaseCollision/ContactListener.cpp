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

#include <SofaBaseCollision/ContactListener.h>

#include <sofa/core/collision/NarrowPhaseDetection.h>
#include <sofa/core/collision/ContactManager.h>

#include <sofa/core/CollisionModel.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/CollisionBeginEvent.h>
#include <sofa/simulation/CollisionEndEvent.h>


namespace sofa
{
namespace core
{

namespace collision
{

SOFA_DECL_CLASS(ContactListener);
int ContactListenerClass = core::RegisterObject("ContactListener .. ").add< ContactListener >();



ContactListener::ContactListener(  CollisionModel* collModel1 , CollisionModel* collModel2 )
    :
      //mLinkCollisionModel1( initLink("collisionModel1", "first collision model"), collModel1 )
      //, mLinkCollisionModel2( initLink("collisionModel2", "second collision model"), collModel2 )
      mNarrowPhase(NULL)
{
    mCollisionModel1 = collModel1;
    mCollisionModel2 = collModel2;
}

ContactListener::~ContactListener()
{
}

void ContactListener::init(void)
{
    helper::vector<ContactManager*> contactManagers;

    mNarrowPhase = getContext()->get<core::collision::NarrowPhaseDetection>();
    if ( mNarrowPhase != NULL )
    {
        // add to the event listening
        f_listening.setValue(true);

    }

}

void ContactListener::handleEvent( core::objectmodel::Event* _event )
{
    if (simulation::CollisionBeginEvent::checkEventType(_event))
    {
        mContactsVector.clear();
    }

    else if (simulation::CollisionEndEvent::checkEventType(_event))
    {
        const NarrowPhaseDetection::DetectionOutputMap& detectionOutputsMap = mNarrowPhase->getDetectionOutputs();
        if ( detectionOutputsMap.size() == 0 )
        {
            endContact(NULL);
            return;
        }

        if  ( mCollisionModel2 == NULL )
        {
            //// check only one collision model
            for (core::collision::NarrowPhaseDetection::DetectionOutputMap::const_iterator it = detectionOutputsMap.begin(); it!=detectionOutputsMap.end(); ++it )
            {
                const CollisionModel* collMod1 = it->first.first;
                const CollisionModel* collMod2 = it->first.second;

                if ( mCollisionModel1 == collMod1 || mCollisionModel1 == collMod2 )
                {
                    if ( const helper::vector<DetectionOutput>* contacts = dynamic_cast<helper::vector<DetectionOutput>*>(it->second) )
                    {
                        mContactsVector.push_back( contacts );
                    }
                }
            }
        }
        else
        {
            // check both collision models
            for (core::collision::NarrowPhaseDetection::DetectionOutputMap::const_iterator it = detectionOutputsMap.begin(); it!=detectionOutputsMap.end(); ++it )
            {
                const CollisionModel* collMod1 = it->first.first;
                const CollisionModel* collMod2 = it->first.second;

                if ( (mCollisionModel1==collMod1 && mCollisionModel2==collMod2) || (mCollisionModel1==collMod2 && mCollisionModel2==collMod1) )
                {
                    if ( const helper::vector<DetectionOutput>* contacts = dynamic_cast<helper::vector<DetectionOutput>*>(it->second) )
                    {
                        mContactsVector.push_back( contacts );
                    }
                }
            }
        }
        beginContact(mContactsVector);
    }
}


template<>
bool ContactListener::canCreate(ContactListener*& obj,
                                core::objectmodel::BaseContext* context,
                                core::objectmodel::BaseObjectDescription* arg)
{
    core::CollisionModel* collModel1 = NULL;
    core::CollisionModel* collModel2 = NULL;

    std::string collModelPath1;
    std::string collModelPath2;

    if (arg->getAttribute("collisionModel1"))
        collModelPath1 = arg->getAttribute("collisionModel1");
    else
        collModelPath1 = "";

    context->findLinkDest(collModel1, collModelPath1, NULL);

    if (arg->getAttribute("collisionModel2"))
        collModelPath2 = arg->getAttribute("collisionModel2");
    else
        collModelPath2 = "";

    context->findLinkDest(collModel2, collModelPath2, NULL);
    if (collModel1 == NULL && collModel2 == NULL )
    {
        context->serr << "Creation of " << className(obj) <<
                         " CollisonListener failed because no Collision Model links are found: \"" << collModelPath1
                      << "\" and \"" << collModelPath2 << "\" " << context->sendl;
        return false;
    }

    return BaseObject::canCreate(obj, context, arg);
}

template<>
ContactListener::SPtr ContactListener::create(ContactListener* ,
                                              core::objectmodel::BaseContext* context,
                                              core::objectmodel::BaseObjectDescription* arg)
{
    CollisionModel* collModel1 = NULL;
    CollisionModel* collModel2 = NULL;

    std::string collModelPath1;
    std::string collModelPath2;

    if(arg)
    {
        collModelPath1 = arg->getAttribute(std::string("collisionModel1"), NULL );
        collModelPath2 = arg->getAttribute(std::string("collisionModel2"), NULL );

        // now 3 cases
        if ( strcmp( collModelPath1.c_str(),"" ) != 0  )
        {
            context->findLinkDest(collModel1, collModelPath1, NULL);
            if ( strcmp( collModelPath2.c_str(),"" ) != 0 )
            {
                context->findLinkDest(collModel2, collModelPath2, NULL);
            }
        }
        else
        {
            context->findLinkDest(collModel1, collModelPath2, NULL);
        }
    }

    typename ContactListener::SPtr obj = sofa::core::objectmodel::New<ContactListener>(collModel1, collModel2);
    if (context)
    {
        context->addObject(obj);
    }

    if (arg)
    {
        obj->parse(arg);
    }

    return obj;
}


} // namespace collision

} // namespace core

} // namespace sofa
