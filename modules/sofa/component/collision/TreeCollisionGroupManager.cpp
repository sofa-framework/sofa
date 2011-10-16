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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/component/collision/TreeCollisionGroupManager.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/tree/GNode.h>
#include <sofa/core/ObjectFactory.h>


namespace sofa
{

namespace component
{

namespace collision
{

SOFA_DECL_CLASS(TreeCollisionGroupManager);

int TreeCollisionGroupManagerClass = core::RegisterObject("Responsible for gathering colliding objects in the same group, for consistent time integration")
        .add< TreeCollisionGroupManager >()
        ;



simulation::Node* TreeCollisionGroupManager::findCommonParent(simulation::Node *group1, simulation::Node* group2)
{
    simulation::tree::GNode *gnodeGroup1=static_cast<simulation::tree::GNode*>(group1),
                             *gnodeGroup2=static_cast<simulation::tree::GNode*>(group2);
    helper::vector<simulation::tree::GNode*> hierarchyParent;

    gnodeGroup1=static_cast<simulation::tree::GNode*>(gnodeGroup1->getParent());
    while ( gnodeGroup1)
    {
        hierarchyParent.push_back(gnodeGroup1);
        gnodeGroup1=static_cast<simulation::tree::GNode*>(gnodeGroup1->getParent());
    }
    if (hierarchyParent.empty())   return NULL;

    gnodeGroup2=static_cast<simulation::tree::GNode*>(gnodeGroup2->getParent());
    while (gnodeGroup2)
    {
        helper::vector<simulation::tree::GNode*>::iterator it=std::find(hierarchyParent.begin(), hierarchyParent.end(), gnodeGroup2);
        if (it != hierarchyParent.end())
        {
            return gnodeGroup2;
            break;
        }
        gnodeGroup2=static_cast<simulation::tree::GNode*>(gnodeGroup2->getParent());
    }

    return NULL;
}

void TreeCollisionGroupManager::clearGroups(core::objectmodel::BaseContext* /*scene*/)
{
    for (std::set<simulation::Node::SPtr>::iterator it = groupSet.begin(); it!=groupSet.end(); ++it)
    {
        simulation::tree::GNode::SPtr group = sofa::core::objectmodel::SPtr_dynamic_cast<simulation::tree::GNode>(*it);
        if (group) clearGroup(group->parent, group.get());
    }

    groupSet.clear();
    groups.clear();
}


}// namespace collision

} // namespace component

} // namespace Sofa
