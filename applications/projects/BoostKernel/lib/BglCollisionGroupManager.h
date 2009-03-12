/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_COLLISION_BGLCOLLISIONGROUPMANAGER_H
#define SOFA_COMPONENT_COLLISION_BGLCOLLISIONGROUPMANAGER_H

#include <sofa/core/componentmodel/collision/CollisionGroupManager.h>
#include "BglNode.h"
#include <sofa/component/component.h>
#include <set>


namespace sofa
{

namespace component
{

namespace collision
{

class BglCollisionGroupManager : public core::componentmodel::collision::CollisionGroupManager
{
public:
    typedef std::map<simulation::Node*,simulation::Node**> GroupSet;
    GroupSet groupSet;
public:
    BglCollisionGroupManager();

    virtual ~BglCollisionGroupManager();

    virtual void createGroups(core::objectmodel::BaseContext* scene, const sofa::helper::vector<core::componentmodel::collision::Contact*>& contacts);

    virtual void clearGroups(core::objectmodel::BaseContext* scene);

    /** Overload this if yo want to design your collision group, e.g. with a MasterSolver.
    Otherwise, an empty Node is returned.
    The OdeSolver is added afterwards.
    */
    virtual simulation::Node* buildCollisionGroup();

protected:
    virtual simulation::Node* getIntegrationNode(core::CollisionModel* model);

    std::map<Instance,GroupSet> storedGroupSet;

    virtual void changeInstance(Instance inst)
    {
        core::componentmodel::collision::CollisionGroupManager::changeInstance(inst);
        storedGroupSet[instance].swap(groupSet);
        groupSet.swap(storedGroupSet[inst]);
    }

};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
