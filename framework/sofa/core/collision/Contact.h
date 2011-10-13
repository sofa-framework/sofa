/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_CORE_COLLISION_CONTACT_H
#define SOFA_CORE_COLLISION_CONTACT_H

#include <sofa/core/collision/DetectionOutput.h>
#include <sofa/core/collision/Intersection.h>
#include <sofa/core/objectmodel/Context.h>
#include <sofa/helper/Factory.h>

#include <vector>

namespace sofa
{

namespace core
{

namespace collision
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
    ///Destructor
    virtual ~Contact() { }
public:
    /// Get the pair of collision models which are in contact
    virtual std::pair< core::CollisionModel*, core::CollisionModel* > getCollisionModels() = 0;

    /// Set the generic description of a contact point
    virtual void setDetectionOutputs(DetectionOutputVector* outputs) = 0;

    virtual void createResponse(objectmodel::BaseContext* group) = 0;

    virtual void removeResponse() = 0;

    /// Return true if this contact should be kept alive, even if objects are no longer in collision
    virtual bool keepAlive() { return false; }

    /// Control the keepAlive flag of the contact. Note that not all contacts support this method
    virtual void setKeepAlive(bool /* val */) {}

    class Factory : public helper::Factory< std::string, Contact, std::pair<std::pair<core::CollisionModel*,core::CollisionModel*>,Intersection*> >
    {
    public:
        static Factory SOFA_CORE_API *getInstance();

        static Object* CreateObject(Key key, Argument arg)
        {
            return getInstance()->createObject(key, arg);
        }

        static Object* CreateAnyObject(Argument arg)
        {
            return getInstance()->createAnyObject(arg);
        }
    };

    /// Create a new contact given 2 collision elements and an intersection method
    static Contact *Create(const std::string& type, core::CollisionModel* model1, core::CollisionModel* model2, Intersection* intersectionMethod);

    template<class RealContact>
    static void create(RealContact*& obj, std::pair<std::pair<core::CollisionModel*,core::CollisionModel*>,Intersection*> arg)
    {
        typedef typename RealContact::CollisionModel1 RealCollisionModel1;
        typedef typename RealContact::CollisionModel2 RealCollisionModel2;
        typedef typename RealContact::Intersection RealIntersection;
        RealCollisionModel1* model1 = dynamic_cast<RealCollisionModel1*>(arg.first.first);
        RealCollisionModel2* model2 = dynamic_cast<RealCollisionModel2*>(arg.first.second);
        RealIntersection* inter  = dynamic_cast<RealIntersection*>(arg.second);
        // CHANGE(Jeremie A. 2007-12-07): disable automatic swapping of the models, as it brings hard to find bugs where the order does not match the DetectionOutputs...
        // The Intersector class is now modified so that they are swapped to an unique order at the detection phase of the pipeline.
        /* if (model1==NULL || model2==NULL)
        { // Try the other way around
            model1 = dynamic_cast<RealCollisionModel1*>(arg.first.second);
            model2 = dynamic_cast<RealCollisionModel2*>(arg.first.first);
        }
        */
        if (model1==NULL || model2==NULL || inter==NULL) return;
        obj = sofa::core::objectmodel::New<RealContact>(model1, model2, inter).get();
    }

};

} // namespace collision

} // namespace core

} // namespace sofa

#endif
