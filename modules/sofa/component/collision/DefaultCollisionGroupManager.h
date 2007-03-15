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
#ifndef SOFA_COMPONENT_COLLISION_DEFAULTCOLLISIONGROUPMANAGER_H
#define SOFA_COMPONENT_COLLISION_DEFAULTCOLLISIONGROUPMANAGER_H

#include <sofa/core/componentmodel/collision/CollisionGroupManager.h>
#include <sofa/simulation/tree/GNode.h>
#include <set>


namespace sofa
{

namespace component
{

namespace collision
{

class DefaultCollisionGroupManager : public core::componentmodel::collision::CollisionGroupManager
{
public:
    std::set<simulation::tree::GNode*> groupSet;
    std::vector<core::objectmodel::BaseContext*> groupVec;
public:
    DefaultCollisionGroupManager();

    virtual ~DefaultCollisionGroupManager();

    virtual void createGroups(core::objectmodel::BaseContext* scene, const std::vector<core::componentmodel::collision::Contact*>& contacts);

    virtual void clearGroups(core::objectmodel::BaseContext* scene);

    virtual const std::vector<core::objectmodel::BaseContext*>& getGroups() { return groupVec; }

protected:
    virtual simulation::tree::GNode* getIntegrationNode(core::CollisionModel* model);
};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
