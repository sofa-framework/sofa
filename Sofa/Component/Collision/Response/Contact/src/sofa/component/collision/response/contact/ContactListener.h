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
#pragma once

#include <sofa/component/collision/response/contact/config.h>
#include <sofa/type/Vec.h>

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/collision/DetectionOutput.h>
#include <sofa/core/collision/NarrowPhaseDetection.h>

namespace sofa::core
{
    // forward declaration
    class CollisionModel;
} // namespace sofa::core

namespace sofa::component::collision::response::contact
{

// forward declaration
class NarrowPhaseDetection;

class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API ContactListener : public virtual core::objectmodel::BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(ContactListener, core::objectmodel::BaseObject);


    ContactListener(core::CollisionModel* collModel1 = nullptr, core::CollisionModel* collModel2 = nullptr );
    ~ContactListener() override ;

    void init(void) override;


    void handleEvent( core::objectmodel::Event* event ) override;

    // Returns the number of stored contacts.
    sofa::Size getNumberOfContacts() const;

    // Returns the distances between the stored contacts as a vector.
    type::vector<double> getDistances() const;

    // Returns the full ContactsVector
    type::vector<type::vector<core::collision::DetectionOutput>> getContactsVector() const;

    // Returns the contact points in the form of a vector of tuples containing two positive integers and two Vec3.
    // The Vec3 store the X, Y, Z coordinates of the points in contact
    // The integers specify to which collision models the points belong. (e.g. (collModel2, (3., 5., 7.), collModel1, (3.1, 5., 6.9)))
    // TODO: replace the tuple with a struct to avoid forgetting which element refers to what.
    std::vector<std::tuple<unsigned int, sofa::type::Vec3, unsigned int, sofa::type::Vec3>> getContactPoints() const; // model, position, model, position

    // Returns the collision elements in the form of a vector of tuples containing four positive integers.
    // The second and fourth integer represent the id of the collision element in the collision models (from a topology)
    // The first and third integer specify to which collision models the ids belong. (e.g. (collModel2, 58, collModel1, 67))
    // TODO: replace the tuple with a struct to avoid forgetting which element refers to what.
    std::vector<std::tuple<unsigned int, unsigned int, unsigned int, unsigned int>> getContactElements() const; // model, id, model, id

    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        core::CollisionModel* collModel1 = nullptr;
        core::CollisionModel* collModel2 = nullptr;

        std::string collModelPath1;
        std::string collModelPath2;

        if (arg->getAttribute("collisionModel1"))
            collModelPath1 = arg->getAttribute("collisionModel1");
        else
            collModelPath1 = "";

        context->findLinkDest(collModel1, collModelPath1, nullptr);

        if (arg->getAttribute("collisionModel2"))
            collModelPath2 = arg->getAttribute("collisionModel2");
        else
            collModelPath2 = "";

        context->findLinkDest(collModel2, collModelPath2, nullptr);

        if (collModel1 == nullptr && collModel2 == nullptr )
        {
            arg->logError("Data attributes 'collisionModel1' and 'collisionModel2' are not pointing to valid collision models.");
            return false;
        }

        return BaseObject::canCreate(obj, context, arg);
    }

    template<class T>
    static typename T::SPtr create(T* , core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        core::CollisionModel* collModel1 = nullptr;
        core::CollisionModel* collModel2 = nullptr;

        std::string collModelPath1;
        std::string collModelPath2;

        if(arg)
        {
            collModelPath1 = arg->getAttribute(std::string("collisionModel1"), "" );
            collModelPath2 = arg->getAttribute(std::string("collisionModel2"), "" );

            // now 3 cases
            if ( strcmp( collModelPath1.c_str(),"" ) != 0  )
            {
                context->findLinkDest(collModel1, collModelPath1, nullptr);

                if ( strcmp( collModelPath2.c_str(),"" ) != 0 )
                {
                    context->findLinkDest(collModel2, collModelPath2, nullptr);
                }
            }
            else
            {
                context->findLinkDest(collModel1, collModelPath2, nullptr);
            }
        }
        typename T::SPtr obj = sofa::core::objectmodel::New<T>( collModel1, collModel2 );

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

protected:
    const core::CollisionModel* m_CollisionModel1;
    const core::CollisionModel* m_CollisionModel2;

private:
    type::vector<type::vector<core::collision::DetectionOutput>> m_ContactsVector;
    type::vector<type::vector<core::collision::DetectionOutput>> m_ContactsVectorBuffer;
    core::collision::NarrowPhaseDetection* m_NarrowPhase;

    virtual void beginContact(const type::vector<type::vector<core::collision::DetectionOutput>>& ) {}
    virtual void endContact(void*) {}
};

} // namespace sofa::component::collision::response::contact
