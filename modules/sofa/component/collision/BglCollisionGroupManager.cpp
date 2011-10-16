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
#include <sofa/component/collision/BglCollisionGroupManager.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/component/collision/SolverMerger.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/CollisionModel.h>
// #include <sofa/helper/system/config.h>
// #include <string.h>


#include <sofa/simulation/bgl/BglNode.h>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/simulation/common/DeleteVisitor.h>

namespace sofa
{

namespace component
{

namespace collision
{

using core::collision::Contact;

SOFA_DECL_CLASS(BglCollisionGroupManager);

int BglCollisionGroupManagerClass = core::RegisterObject("Responsible for gathering colliding objects in the same group, for consistent time integration")
        .add< BglCollisionGroupManager >()
        ;



simulation::Node* BglCollisionGroupManager::findCommonParent(simulation::Node *group1, simulation::Node* group2)
{
    simulation::bgl::BglNode *bglGroup1=static_cast<simulation::bgl::BglNode*>(group1),
                              *bglGroup2=static_cast<simulation::bgl::BglNode*>(group2);

    typedef std::vector< simulation::Node*> ParentsContainer;

    ParentsContainer pgroup1; bglGroup1->getParents(pgroup1);
    if (pgroup1.empty()) return NULL;


    ParentsContainer pgroup2; bglGroup2->getParents(pgroup2);
    return compatibleSetOfNode(pgroup1, pgroup2);
}

void BglCollisionGroupManager::clearGroups(core::objectmodel::BaseContext* /*scene*/)
{
    for (std::set<simulation::Node::SPtr>::iterator it = groupSet.begin(); it!=groupSet.end(); ++it)
    {
        sofa::simulation::bgl::BglNode::SPtr group = sofa::core::objectmodel::SPtr_dynamic_cast<sofa::simulation::bgl::BglNode>(*it);
        if (group) clearGroup(group->parents, group.get());
    }
    groupSet.clear();
    groups.clear();
}



template <typename ContainerParent>
typename ContainerParent::value_type BglCollisionGroupManager::compatibleSetOfNode( ContainerParent &set1,ContainerParent &set2)
{
    typename ContainerParent::iterator
    it1, it1_end=set1.end(),
         it2, it2_end=set2.end();
    for (it1=set1.begin(); it1!=it1_end; ++it1)
    {
        for (it2=set2.begin(); it2!=it2_end; ++it2)
        {
            if (*it1 == *it2)
            {
                return *it1;
            }
        }
    }
    return NULL;
}


}// namespace collision

} // namespace component

} // namespace Sofa
