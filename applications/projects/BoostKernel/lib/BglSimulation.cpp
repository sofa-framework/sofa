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

#include "BuildNodesFromGNodeVisitor.h"
#include "BuildRestFromGNodeVisitor.h"



#include <sofa/simulation/tree/TreeSimulation.h>

#include <sofa/simulation/common/Visitor.h>
#include <sofa/simulation/common/AnimateVisitor.h>
#include <sofa/simulation/common/BehaviorUpdatePositionVisitor.h>
#include <sofa/simulation/common/CleanupVisitor.h>
#include <sofa/simulation/common/DeleteVisitor.h>
#include <sofa/simulation/common/InitVisitor.h>
#include <sofa/simulation/common/PrintVisitor.h>
#include <sofa/simulation/common/PropagateEventVisitor.h>
#include <sofa/simulation/common/UpdateMappingVisitor.h>
#include <sofa/simulation/common/VisualVisitor.h>

#include <sofa/simulation/common/AnimateBeginEvent.h>
#include <sofa/simulation/common/AnimateEndEvent.h>
#include <sofa/simulation/common/UpdateMappingEndEvent.h>


#include <sofa/helper/system/FileRepository.h>

#include <iostream>
#include <algorithm>

#include "BglNode.h"
using std::cerr;
using std::endl;

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

BglSimulation::BglSimulation():
    solverEulerEuler(NULL),
    solverRungeKutta4RungeKutta4(NULL),
    solverCGImplicitCGImplicit(NULL),
    solverEulerImplicitEulerImplicit(NULL),
    solverStaticSolver(NULL),
    solverRungeKutta4Euler(NULL),
    solverCGImplicitEuler(NULL),
    solverCGImplicitRungeKutta4(NULL),
    solverEulerImplicitEuler(NULL),
    solverEulerImplicitRungeKutta4(NULL),
    solverEulerImplicitCGImplicit(NULL)
{

}




/// Create a graph node and attach a new Node to it, then return the Node
Node* BglSimulation::newNode(const std::string& name)
{
    return graphManager.newNode(name);
}


//
//      void BglSimulation::setSolverOfCollisionGroup(Node* solverNode, Node* solverOfCollisionGroup)
//      {
//        solver_colisionGroup_map[solverNode] = solverOfCollisionGroup;
//        nodeGroupSolvers.insert(solverOfCollisionGroup);
////         std::cerr << "Inserting " << solverOfCollisionGroup->getName() << "\n";
//      }


/**
Data: hgraph, rgraph
 Result: hroots, interaction groups, all nodes initialized.
    */
void BglSimulation::init(Node* root )
{
    Simulation::init(root);

//         /// compute the interaction groups
    graphManager.computeInteractionGraphAndConnectedComponents();

}




/// TODO: adapt the AnimateVisitor to BGL
void BglSimulation::animate(Node* root, double dt)
{
    dt = root->getContext()->getDt();

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printComment(std::string("Begin Step"));
#endif
    Node *masterNode = graphManager.getMasterNode();

    {
        AnimateBeginEvent ev ( dt );
        PropagateEventVisitor act ( &ev );
        masterNode->doExecuteVisitor ( &act );
    }


    double startTime = root->getTime();
    double mechanicalDt = dt/numMechSteps.getValue();

    for( unsigned step=0; step<numMechSteps.getValue(); step++ )
    {
        graphManager.collisionStep(root,dt);
        graphManager.mechanicalStep(root,dt);
        graphManager.clearMasterVertex();
        graphManager.insertHierarchicalGraph();

        BehaviorUpdatePositionVisitor beh(dt);
        masterNode->doExecuteVisitor ( &beh );
        masterNode->setTime ( startTime + (step+1)* mechanicalDt );
    }

    {
        AnimateEndEvent ev ( dt );
        PropagateEventVisitor act ( &ev );
        masterNode->doExecuteVisitor( &act );
    }

    //Update Mapping
    {
        UpdateMappingVisitor actMapping;
        masterNode->doExecuteVisitor( &actMapping);
        simulation::UpdateMappingEndEvent ev ( dt );
        PropagateEventVisitor act ( &ev );
        masterNode->doExecuteVisitor( &act );
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printComment(std::string("End Step"));
#endif

}

void BglSimulation::computeBBox(Node* /*root*/, SReal* minBBox, SReal* maxBBox)
{
    sofa::simulation::Simulation::computeBBox(graphManager.getMasterNode(),minBBox,maxBBox);
}


void BglSimulation::draw(Node* masterNode, helper::gl::VisualParameters*)
{
    if (!masterNode) return;

    masterNode->glDraw();

}


/// Create a GNode tree structure using available file loaders, then convert it to a BglSimulation
Node* BglSimulation::load(const char* f)
{
    graphManager.reset();
    std::string fileName(f);
    /*  if (fileName.empty())*/
    {
        //         fileName = "liver.scn";
        sofa::helper::system::DataRepository.findFile(fileName);
    }

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

    std::map<simulation::Node*,BglNode*> gnode_bnode_map;
    BuildNodesFromGNodeVisitor b1(this);
    groot->execute(b1);


    gnode_bnode_map = b1.getGNodeBNodeMap();
    BuildRestFromGNodeVisitor b2(&graphManager);
    b2.setGNodeBNodeMap(gnode_bnode_map);
    groot->execute(b2);


    Node *masterNode=graphManager.getMasterNode();
    const sofa::core::objectmodel::Context &c = *( (sofa::core::objectmodel::Context*)groot->getContext());
    masterNode->copyContext(c);

    graphManager.update();

    /// find the roots in hgraph
    graphManager.computeRoots();

    graphManager.insertHierarchicalGraph();

    return masterNode;
}

void BglSimulation::clear()
{
    graphManager.clear();
}

void BglSimulation::reset(Node* root)
{
    sofa::simulation::Simulation::reset(root);
//         sofa::simulation::Simulation::reset(visualNode);

    graphManager.update();
    graphManager.reset();

//        instruments.clear();
//        instrumentInUse.setValue(-1);
}

void BglSimulation::unload(Node* root)
{
    if (!root) return;
    root->execute<CleanupVisitor>();
    DeleteVisitor deleteGraph;
    graphManager.getMasterNode()->doExecuteVisitor(&deleteGraph);
    clear();
}

Node* BglSimulation::getSolverEulerEuler()
{
    if (!solverEulerEuler)
    {
        solverEulerEuler = newNode("SolverEulerEuler");
    }
    return solverEulerEuler;
}
Node* BglSimulation::getSolverRungeKutta4RungeKutta4()
{
    if (!solverRungeKutta4RungeKutta4)
    {
        solverRungeKutta4RungeKutta4 = newNode("SolverRungeKutta4RungeKutta4");
    }
    return solverRungeKutta4RungeKutta4;
}
Node* BglSimulation::getSolverCGImplicitCGImplicit()
{
    if (!solverCGImplicitCGImplicit)
    {
        solverCGImplicitCGImplicit = newNode("SolverCGImplicitCGImplicit");
    }
    return solverCGImplicitCGImplicit;
}
Node* BglSimulation::getSolverEulerImplicitEulerImplicit()
{
    if (!solverEulerImplicitEulerImplicit)
    {
        solverEulerImplicitEulerImplicit = newNode("SolverEulerImplicitEulerImplicit");
    }
    return solverEulerImplicitEulerImplicit;
}
Node* BglSimulation::getSolverStaticSolver()
{
    if (!solverStaticSolver)
    {
        solverStaticSolver = newNode("SolverStaticSolver");
    }
    return solverStaticSolver;
}
Node* BglSimulation::getSolverRungeKutta4Euler()
{
    if (!solverRungeKutta4Euler)
    {
        solverRungeKutta4Euler = newNode("SolverRungeKutta4Euler");
    }
    return solverRungeKutta4Euler;
}
Node* BglSimulation::getSolverCGImplicitEuler()
{
    if (!solverCGImplicitEuler)
    {
        solverCGImplicitEuler = newNode("SolverCGImplicitEuler");
    }
    return solverCGImplicitEuler;
}
Node* BglSimulation::getSolverCGImplicitRungeKutta4()
{
    if (!solverCGImplicitRungeKutta4)
    {
        solverCGImplicitRungeKutta4 = newNode("SolverCGImplicitRungeKutta4");
    }
    return solverCGImplicitRungeKutta4;
}
Node* BglSimulation::getSolverEulerImplicitEuler()
{
    if (!solverEulerImplicitEuler)
    {
        solverEulerImplicitEuler = newNode("SolverEulerImplicitEuler");
    }
    return solverEulerImplicitEuler;
}
Node* BglSimulation::getSolverEulerImplicitRungeKutta4()
{
    if (!solverEulerImplicitRungeKutta4)
    {
        solverEulerImplicitRungeKutta4 = newNode("SolverEulerImplicitRungeKutta4");
    }
    return solverEulerImplicitRungeKutta4;
}
Node* BglSimulation::getSolverEulerImplicitCGImplicit()
{
    if (!solverEulerImplicitCGImplicit)
    {
        solverEulerImplicitCGImplicit = newNode("SolverEulerImplicitCGImplicit");
    }
    return solverEulerImplicitCGImplicit;
}

}
}
}


