/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_CORE_COMPONENTMODEL_COLLISION_CONTACT_H
#define SOFA_CORE_COMPONENTMODEL_COLLISION_CONTACT_H

#include <sofa/core/componentmodel/collision/DetectionOutput.h>
#include <sofa/core/componentmodel/collision/Intersection.h>
#include <sofa/core/objectmodel/Context.h>
#include <sofa/helper/Factory.h>

#include <vector>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace collision
{

using namespace sofa::defaulttype;

class Contact : public virtual objectmodel::BaseObject
{
public:
    typedef Intersection::DetectionOutputVector DetectionOutputVector;

    virtual ~Contact() { }

    virtual std::pair< core::CollisionModel*, core::CollisionModel* > getCollisionModels() = 0;

    virtual void setDetectionOutputs(DetectionOutputVector& outputs) = 0;

    virtual void createResponse(objectmodel::BaseContext* group) = 0;

    virtual void removeResponse() = 0;

    /// Return true if this contact should be kept alive, even if objects are no longer in collision
    virtual bool keepAlive() { return false; }

    typedef helper::Factory< std::string, Contact, std::pair<std::pair<core::CollisionModel*,core::CollisionModel*>,Intersection*> > Factory;

    static Contact* Create(const std::string& type, core::CollisionModel* model1, core::CollisionModel* model2, Intersection* intersectionMethod);
};

template<class RealContact>
void create(RealContact*& obj, std::pair<std::pair<core::CollisionModel*,core::CollisionModel*>,Intersection*> arg)
{
    typedef typename RealContact::CollisionModel1 RealCollisionModel1;
    typedef typename RealContact::CollisionModel2 RealCollisionModel2;
    typedef typename RealContact::Intersection RealIntersection;
    RealCollisionModel1* model1 = dynamic_cast<RealCollisionModel1*>(arg.first.first);
    RealCollisionModel2* model2 = dynamic_cast<RealCollisionModel2*>(arg.first.second);
    RealIntersection* inter  = dynamic_cast<RealIntersection*>(arg.second);
    if (model1==NULL || model2==NULL)
    {
        // Try the other way around
        model1 = dynamic_cast<RealCollisionModel1*>(arg.first.second);
        model2 = dynamic_cast<RealCollisionModel2*>(arg.first.first);
    }
    if (model1==NULL || model2==NULL || inter==NULL) return;
    obj = new RealContact(model1, model2, inter);
}

} // namespace collision

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
