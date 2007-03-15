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
#ifndef SOFA_CORE_COMPONENTMODEL_COLLISION_COLLISIONGROUPMANAGER_H
#define SOFA_CORE_COMPONENTMODEL_COLLISION_COLLISIONGROUPMANAGER_H

#include <sofa/core/componentmodel/collision/Contact.h>
#include <sofa/core/objectmodel/BaseObject.h>

#include <vector>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace collision
{

//class Scene;

class CollisionGroupManager : public virtual objectmodel::BaseObject
{
public:
    virtual ~CollisionGroupManager() { }

    virtual void createGroups(objectmodel::BaseContext* scene, const std::vector<Contact*>& contacts) = 0;

    virtual void clearGroups(objectmodel::BaseContext* scene) = 0;

    virtual const std::vector<objectmodel::BaseContext*>& getGroups() = 0;
};

} // namespace collision

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
