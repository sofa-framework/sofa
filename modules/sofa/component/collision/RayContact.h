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
#ifndef SOFA_COMPONENT_COLLISION_RAYCONTACT_H
#define SOFA_COMPONENT_COLLISION_RAYCONTACT_H

#include <sofa/core/componentmodel/collision/Contact.h>
#include <sofa/helper/Factory.h>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;

class RayModel;

class BaseRayContact : public core::componentmodel::collision::Contact
{
public:
    typedef RayModel CollisionModel1;

protected:
    CollisionModel1* model1;
    std::vector<core::componentmodel::collision::DetectionOutput*> collisions;

public:
    BaseRayContact(CollisionModel1* model1, core::componentmodel::collision::Intersection* instersectionMethod);

    ~BaseRayContact();

    void setDetectionOutputs(const std::vector<core::componentmodel::collision::DetectionOutput*>& outputs)
    {
        collisions = outputs;
    }

    const std::vector<core::componentmodel::collision::DetectionOutput*>& getDetectionOutputs() const { return collisions; }

    void createResponse(core::objectmodel::BaseContext* /*group*/)
    {
    }

    void removeResponse()
    {
    }
};

template<class CM2>
class RayContact : public BaseRayContact
{
public:
    typedef RayModel CollisionModel1;
    typedef CM2 CollisionModel2;
    typedef core::componentmodel::collision::Intersection Intersection;
protected:
    CollisionModel2* model2;
    core::objectmodel::BaseContext* parent;
public:
    RayContact(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod)
        : BaseRayContact(model1, intersectionMethod), model2(model2)
    {
    }

    std::pair<core::CollisionModel*,core::CollisionModel*> getCollisionModels() { return std::make_pair(model1,model2); }
};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
