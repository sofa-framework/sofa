/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
//
// C++ Implementation: BglSimulation
//
// Description:
//
//
// Author: Francois Faure in The SOFA team </www.sofa-framework.org>, (C) 2008
//
// Copyright: See COPYING file that comes with this distribution
//
//
#include <sofa/simulation/bgl/BglSimulation.h>
#include <sofa/simulation/bgl/BglNode.h>

#include <sofa/core/ObjectFactory.h>

namespace sofa
{
namespace simulation
{
namespace bgl
{


Simulation* getSimulation()
{
    if ( simulation::Simulation::theSimulation.get() == 0 )
        setSimulation(new BglSimulation);
    return simulation::getSimulation();
}


BglSimulation::BglSimulation()
{
    //-------------------------------------------------------------------------------------------------------
    sofa::core::ObjectFactory::AddAlias("DefaultCollisionGroupManager",
            "BglCollisionGroupManager", true, 0);

    sofa::core::ObjectFactory::AddAlias("CollisionGroup",
            "BglCollisionGroupManager", true, 0);

    sofa::simulation::xml::BaseElement::NodeFactory::DuplicateEntry("BglNodeMultiMapping","MultiMappingObject");
}


/// Create a graph node and attach a new Node to it, then return the Node
Node::SPtr BglSimulation::createNewGraph(const std::string& name)
{
    return sofa::core::objectmodel::New<BglNode>(name);
}

/**
Data: hgraph, rgraph
 Result: hroots, interaction groups, all nodes initialized.
    */
void BglSimulation::init(Node* root )
{
    Simulation::init(root);
    BglGraphManager::getInstance()->update();
}


/// Create a GNode tree structure using available file loaders, then convert it to a BglSimulation
Node::SPtr BglSimulation::load(const char* f)
{
    Node::SPtr root=Simulation::load(f);
    BglGraphManager::getInstance()->update();
    return root;

    //Temporary: we need to change that: We could change getRoots by a getRoot.
    //if several roots are found, we return a master node, above the roots of the simulation

//         std::vector< Node* > roots;
//         BglGraphManager::getInstance()->getRoots(roots);
//         if (roots.empty()) return NULL;
//         return roots.back();
}


void BglSimulation::reset(Node* root)
{
    sofa::simulation::Simulation::reset(root);
    BglGraphManager::getInstance()->reset();
}

void BglSimulation::unload(Node::SPtr root)
{
    BglNode::SPtr n=sofa::core::objectmodel::SPtr_dynamic_cast<BglNode>(root);
    if (!n) return;
    helper::vector< Node* > parents;
    n->getParents(parents);
    if (parents.empty()) //Root
    {
        /// TODO: BglSimulation graphs are never deleted ?
        //Simulation::unload(root);
    }
}
}
}
}


