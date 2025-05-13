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

#include <sofa/core/collision/DetectionOutput.h>
#include <sofa/core/collision/Intersection.h>
#include <sofa/core/objectmodel/Context.h>
#include <sofa/helper/Factory.h>

#include <vector>

namespace sofa::core::collision
{
/**
 * @brief contact response component handling the response between a pair of models
 *
 * - Dynamically created by the ContactManager
 *
 *   -# Persistent between iterations
 *
 *   -# New id data in DetectionOutput allow to keep an history of a contact
 *
 * - In most cases : create and initialize the real response component
 *
 *   -#InteractionForceField, Constraint, ...
 *
 * - Contact object dynamically appears in the scenegraph
 */
class SOFA_CORE_API Contact : public virtual objectmodel::BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(Contact, objectmodel::BaseObject);
protected:
    Contact() {}
    ///Destructor
    ~Contact() override { }
	
private:
    Contact(const Contact& n) = delete;
    Contact& operator=(const Contact& n) = delete;
	
public:
    /**
     * !!! WARNING since v25.12 !!! 
     * 
     * The template method pattern has been applied to this part of the API. 
     * This method calls the newly introduced method "doGetCollisionModels" internally,
     * which is the method to override from now on.
     *
     * Get the pair of collision models which are in contact 
     * 
     **/
    virtual std::pair< core::CollisionModel*, core::CollisionModel* > getCollisionModels() final
    {
        //TODO (SPRINT SED 2025): Component state mechamism
        return this->doGetCollisionModels();
    };

    /**
     * !!! WARNING since v25.12 !!! 
     * 
     * The template method pattern has been applied to this part of the API. 
     * This method calls the newly introduced method "doSetDetectionOuputs" internally,
     * which is the method to override from now on.
     *
     * Set the generic description of a contact point
     * 
     **/
    virtual void setDetectionOutputs(DetectionOutputVector* outputs) final 
    {
        //TODO (SPRINT SED 2025): Component state mechamism
        this->doSetDetectionOutputs(outputs);
    };

    /**
     * !!! WARNING since v25.12 !!! 
     * 
     * The template method pattern has been applied to this part of the API. 
     * This method calls the newly introduced method "doCreateResponse" internally,
     * which is the method to override from now on.
     *
     **/
    virtual void createResponse(objectmodel::BaseContext* group) final
    {
        //TODO (SPRINT SED 2025): Component state mechamism
        this->doCreateResponse(group);
    };

    /**
     * !!! WARNING since v25.12 !!! 
     * 
     * The template method pattern has been applied to this part of the API. 
     * This method calls the newly introduced method "doRemoveResponse" internally,
     * which is the method to override from now on.
     *
     **/
    virtual void removeResponse() final
    {
        //TODO (SPRINT SED 2025): Component state mechamism
        this->doRemoveResponse();
    };

    /// Return true if this contact should be kept alive, even if objects are no longer in collision
    virtual bool keepAlive() { return false; }

    /// Control the keepAlive flag of the contact. Note that not all contacts support this method
    virtual void setKeepAlive(bool /* val */) {}

    //Todo adding TPtr parameter
    class SOFA_CORE_API Factory : public helper::Factory< std::string, Contact, std::pair<std::pair<core::CollisionModel*,core::CollisionModel*>,Intersection*>, Contact::SPtr >
    {
    public:
        static Factory *getInstance();

        static ObjectPtr CreateObject(Key key, Argument arg)
        {
            return getInstance()->createObject(key, arg);
        }

        static ObjectPtr CreateAnyObject(Argument arg)
        {
            return getInstance()->createAnyObject(arg);
        }
    };

    /// Create a new contact given 2 collision elements and an intersection method
    static Contact::SPtr Create(const std::string& type, core::CollisionModel* model1, core::CollisionModel* model2, Intersection* intersectionMethod, bool verbose=true);

    template<class RealContact>
    static typename RealContact::SPtr create(RealContact*, std::pair<std::pair<core::CollisionModel*,core::CollisionModel*>,Intersection*> arg)
    {
        typedef typename RealContact::CollisionModel1 RealCollisionModel1;
        typedef typename RealContact::CollisionModel2 RealCollisionModel2;
        typedef typename RealContact::Intersection RealIntersection;
        RealCollisionModel1* model1 = dynamic_cast<RealCollisionModel1*>(arg.first.first);
        RealCollisionModel2* model2 = dynamic_cast<RealCollisionModel2*>(arg.first.second);
        RealIntersection* inter  = dynamic_cast<RealIntersection*>(arg.second);
        if (model1==nullptr || model2==nullptr || inter==nullptr) return typename RealContact::SPtr();
        return sofa::core::objectmodel::New<RealContact>(model1, model2, inter);
    }

protected:
    /// Get the pair of collision models which are in contact
    virtual std::pair< core::CollisionModel*, core::CollisionModel* > doGetCollisionModels() = 0;

    /// Set the generic description of a contact point
    virtual void doSetDetectionOutputs(DetectionOutputVector* outputs) = 0;

    virtual void doCreateResponse(objectmodel::BaseContext* group) = 0;

    virtual void doRemoveResponse() = 0;


};
} // namespace sofa::core::collision
