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
#ifndef SOFA_CORE_COMPONENTMODEL_COLLISION_DETECTION_H
#define SOFA_CORE_COMPONENTMODEL_COLLISION_DETECTION_H

#include <sofa/core/CollisionModel.h>
#include <sofa/core/componentmodel/collision/Intersection.h>
#include <vector>
#include <algorithm>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace collision
{

class Detection : public virtual objectmodel::BaseObject
{
protected:
    /// Current intersection method
    Intersection* intersectionMethod;

    /// Contains the collisions models
    /// which are included in the broadphase
    /// but which are not in collisions with another model
    std::vector<core::CollisionModel*> cmNoCollision;
public:

    Detection()
        : intersectionMethod(NULL)
    {
    }

    /// virtual because subclasses might do precomputations based on intersection algorithms
    virtual void setIntersectionMethod(Intersection* v) { intersectionMethod = v;    }
    Intersection* getIntersectionMethod() const         { return intersectionMethod; }

    void removeCmNoCollision(core::CollisionModel* cm)
    {
        std::vector<core::CollisionModel*>::iterator it = std::find(cmNoCollision.begin(), cmNoCollision.end(), cm);
        if (it != cmNoCollision.end())
        {
            cmNoCollision.erase(it);
        }
    }

    void addNoCollisionDetect (core::CollisionModel* cm)
    {
        cmNoCollision.push_back(cm);
    }

    std::vector<core::CollisionModel*>& getListNoCollisionModel() {return cmNoCollision;};
};

} // namespace collision

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
