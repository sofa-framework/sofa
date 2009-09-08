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
#include "BglSimulation.h"
#include "BglNode.h"

#include "BuildNodesFromGNodeVisitor.h"
#include "BuildRestFromGNodeVisitor.h"

#include "BglGraphManager.inl"

#include <sofa/simulation/tree/TreeSimulation.h>
#include <sofa/simulation/common/AnimateBeginEvent.h>
#include <sofa/simulation/common/PropagateEventVisitor.h>
#include <sofa/simulation/common/BehaviorUpdatePositionVisitor.h>
#include <sofa/simulation/common/UpdateContextVisitor.h>
#include <sofa/simulation/common/AnimateEndEvent.h>
#include <sofa/simulation/common/UpdateMappingEndEvent.h>
#include <sofa/simulation/common/UpdateMappingVisitor.h>
#include <sofa/simulation/common/VisualVisitor.h>



namespace sofa
{
namespace simulation
{
namespace bgl
{


Simulation* getSimulation()
{
    if ( Simulation::Simulation::theSimulation==NULL )
        setSimulation(new BglSimulation);
    return simulation::getSimulation();
}

BglSimulation::BglSimulation()
{
}




/// Create a graph node and attach a new Node to it, then return the Node
Node* BglSimulation::newNode(const std::string& name)
{
    return new BglNode(name);
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
Node* BglSimulation::load(const char* f)
{
    std::string fileName(f);
    sofa::helper::system::DataRepository.findFile(fileName);

    sofa::simulation::tree::GNode* groot = 0;

    sofa::simulation::tree::TreeSimulation treeSimu;
    std::string in_filename(fileName);
    if (in_filename.rfind(".simu") == std::string::npos)
        groot = dynamic_cast< sofa::simulation::tree::GNode* >(treeSimu.load(fileName.c_str()));

    if ( !groot )
    {
        cerr<<"BglSimulation::load file "<<fileName<<" failed"<<endl;
        exit(1);
    }

    BuildNodesFromGNodeVisitor b1(this);
    groot->execute(b1);


    std::map<simulation::Node*,BglNode*> gnode_bnode_map;
    gnode_bnode_map = b1.getGNodeBNodeMap();
    BuildRestFromGNodeVisitor b2;
    b2.setGNodeBNodeMap(gnode_bnode_map);
    groot->execute(b2);


    BglGraphManager::getInstance()->update();
    std::vector< Node* > roots;
    BglGraphManager::getInstance()->getRoots(roots);


    //Temporary: we need to change that: We could change getRoots by a getRoot.
    //if several roots are found, we return a master node, above the roots of the simulation
    if (roots.empty()) return NULL;
    return roots.back();
}


void BglSimulation::reset(Node* root)
{
    sofa::simulation::Simulation::reset(root);
    BglGraphManager::getInstance()->reset();
}

void BglSimulation::unload(Node* root)
{
    if (!root) return;
    Simulation::unload(root);
    delete root;
    clear();
}

void BglSimulation::clear()
{
    BglGraphManager::getInstance()->clear();
}
}
}
}


