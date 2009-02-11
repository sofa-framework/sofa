/******************************************************************************
 *       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
 *                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include "BglNode.h"
#include "BglSimulation.h"
#include "dfs_adapter.h"
#include "dfv_adapter.h"
#include "BuildNodesFromGNodeVisitor.h"
#include "BuildRestFromGNodeVisitor.h"
#include "BglDeleteVisitor.h"

#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/vector_property_map.hpp>

#include <sofa/simulation/tree/TreeSimulation.h>

#include <sofa/simulation/common/Visitor.h>
#include <sofa/simulation/common/AnimateVisitor.h>
#include <sofa/simulation/common/BehaviorUpdatePositionVisitor.h>
#include <sofa/simulation/common/CollisionVisitor.h>
#include <sofa/simulation/common/CleanupVisitor.h>
#include <sofa/simulation/common/PrintVisitor.h>
#include <sofa/simulation/common/PropagateEventVisitor.h>
#include <sofa/simulation/common/VisualVisitor.h>

#include <sofa/simulation/common/AnimateBeginEvent.h>
#include <sofa/simulation/common/AnimateEndEvent.h>
#include <sofa/simulation/common/UpdateMappingEndEvent.h>


#include <sofa/helper/system/FileRepository.h>

#include <iostream>
#include <algorithm>
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
    collisionPipeline(NULL),
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

    h_vertex_node_map = get( bglnode_t(), hgraph);
    r_vertex_node_map = get( bglnode_t(), rgraph);
    // 	c_vertex_node_map = get( bglnode_t(), cgraph);

    // The animation control overloads the solvers of the scene
    masterNode= static_cast<BglNode*>(newNodeNow("masterNode"));
    masterVertex = h_node_vertex_map[masterNode];

    // The collision Node
    collisionNode= static_cast<BglNode*>(newNodeNow("collisionNode"));
    collisionVertex = h_node_vertex_map[collisionNode];

}

// BglSimulation::Hvertex BglSimulation::addHvertex( sofa::simulation::Node* node )
// {
//     Hvertex hvertex =  add_vertex( hgraph);
//     h_vertex_node_map[hvertex] = node;
//      h_node_vertex_map[node] = hvertex;
//
//      Rvertex rvertex =  add_vertex( rgraph);
//     r_vertex_node_map[rvertex] = node;
//      r_node_vertex_map[node] = rvertex;
//
//      return node;
// }

BglSimulation::Rvertex BglSimulation::convertHvertex2Rvertex(Hvertex v)
{
    return r_node_vertex_map[ h_vertex_node_map[ v ] ];
}
BglSimulation::Hvertex BglSimulation::convertRvertex2Hvertex(Rvertex v)
{
    return h_node_vertex_map[ r_vertex_node_map[ v ] ];
}
void BglSimulation::addEdge( Hvertex p, Hvertex c )
{
    if (p == c) return;
//         std::cerr << "#####addEdge : " << h_vertex_node_map[p]->getName() << " : "
//                   <<  h_vertex_node_map[c]->getName() << "@"<< h_vertex_node_map[c]  << "\n";
    addHedge(p,c);
    addRedge(convertHvertex2Rvertex(c),convertHvertex2Rvertex(p));
}
void BglSimulation::removeEdge( Hvertex p, Hvertex c )
{
    if (p == c) return;
//         std::cerr << "#####removeEdge : " << h_vertex_node_map[p]->getName() << " : "
//                   <<  h_vertex_node_map[c]->getName() << "@"<< h_vertex_node_map[c] << "\n";
    removeHedge(p,c);
    removeRedge(convertHvertex2Rvertex(c),convertHvertex2Rvertex(p));
}

void BglSimulation::deleteVertex( Hvertex v)
{
//         std::cerr << "#####removeEdge : "
//                   << h_vertex_node_map[v]->getName() << "@"<< h_vertex_node_map[v]<< "\n";
    deleteHvertex(v);
    deleteRvertex(convertHvertex2Rvertex(v));
}

void BglSimulation::clearVertex( Hvertex v )
{
    clear_vertex(v,hgraph);
    clear_vertex(convertHvertex2Rvertex(v),rgraph);
}


BglSimulation::Hedge BglSimulation::addHedge( Hvertex p, Hvertex c )
{
//         std::cerr << "addHedge : " << h_vertex_node_map[p]->getName() << " : " <<  h_vertex_node_map[c]->getName() << "\n";
    std::pair<Hedge, bool> e =  add_edge(  p,c,hgraph);
    assert(e.second);
    return e.first;
}

BglSimulation::Redge BglSimulation::addRedge( Rvertex p, Rvertex c )
{
//         std::cerr << "addRedge : " << r_vertex_node_map[p]->getName() << " : " <<  r_vertex_node_map[c]->getName() << "\n";
    std::pair<Redge, bool> e =  add_edge(  p,c,rgraph);
    assert(e.second);
    return e.first;
}


void BglSimulation::removeHedge( Hvertex p, Hvertex c )
{
//         std::cerr << "removeHedge : " << h_vertex_node_map[p]->getName() << " : " <<  h_vertex_node_map[c]->getName() << "\n";
    remove_edge(  p,c,hgraph);
}

void BglSimulation::removeRedge( Rvertex p, Rvertex c )
{
//         std::cerr << "removeRedge : " << r_vertex_node_map[p]->getName() << " : " <<  r_vertex_node_map[c]->getName() << "\n";
    remove_edge(  p,c,rgraph);
}



/// Create a graph node and attach a new Node to it, then return the Node
Node* BglSimulation::newNode(const std::string& name)
{
//         std::cerr << "new Node : " << name << "\n";
    BglNode* s  = new BglNode(this,name);
    nodeToAdd.push_back(s);
//         std::cerr << "\t @" << s << "\n";
    return s;
}

void BglSimulation::insertNewNode(Node *n)
{
//         std::cerr << "Effectively add " << n->getName() << "\n";
    BglNode *bglN = static_cast<BglNode*>(n);
    //Effectively create a vertex in the graph
    Hvertex hnode=add_vertex(hgraph);
    h_vertex_node_map[hnode] = bglN;
    h_node_vertex_map[bglN] = hnode;

    //Reverse graph
    Rvertex rnode = add_vertex(rgraph);
    r_vertex_node_map[rnode] = bglN;
    r_node_vertex_map[bglN] = rnode;

    //configure the node
    bglN->graph = &hgraph;
    bglN->vertexId = hnode;
}


Node* BglSimulation::newNodeNow(const std::string& name)
{
//          std::cerr << "new Node : " << name << "\n";
    // Each BglNode needs a vertex in hgraph
    Hvertex hnode =  add_vertex( hgraph);
    BglNode* s  = new BglNode(this,&hgraph,hnode,name);
    h_vertex_node_map[hnode] = s;
    h_node_vertex_map[s] = hnode;
    // add it to rgraph
    Rvertex rnode = add_vertex( rgraph );
    r_vertex_node_map[rnode] = s;
    r_node_vertex_map[s] = rnode;
//         std::cerr << "\t @" << s << "\n";
    return s;
}


/// Add a node to as the child of another
void BglSimulation::addNode(BglNode* parent, BglNode* child)
{
//           std::cerr << "ADD Node " << parent->getName() << "-->" << child->getName() << " : \n";
//           if (parent != masterNode)

//           std::cerr << "Add Node : push an edge " << parent->getName() << "-->" << child->getName() << " : \n";
    edgeToAdd.insert(std::make_pair(parent,child));
}
void BglSimulation::deleteNode( Node* n)
{
//         std::cerr << "Delete Node " << n << " : \n";
    vertexToDelete.insert(h_node_vertex_map[n]);
}


void BglSimulation::deleteHvertex( Hvertex vH)
{
//         std::cerr << "Delete Now Node " << n << " : \n";
//         std::cerr << n->getName() << " : " << h_node_vertex_map[n] << "\n";
    Node *node = h_vertex_node_map[vH];
    clear_vertex(vH, hgraph);
    h_node_vertex_map.erase(node);

    //Cannot remove vertex: it changes all the vertices of the graph !!!
//         remove_vertex(vH, hgraph);
}


void BglSimulation::deleteRvertex( Rvertex vR)
{
//         std::cerr << "Delete Now Node " << n << " : \n";
//         std::cerr << n->getName() << " : " << h_node_vertex_map[n] << "\n";
    Node *node = r_vertex_node_map[vR];
    clear_vertex(vR, rgraph);
    r_node_vertex_map.erase(node);

    //Cannot remove vertex: it changes all the vertices of the graph !!!
//         remove_vertex(vR, rgraph);
}



/// Method called when a MechanicalMapping is created.
void BglSimulation::setMechanicalMapping(Node* , core::componentmodel::behavior::BaseMechanicalMapping* m )
{
    Node *from=(Node*)m->getMechFrom()->getContext();
    Node *to=(Node*)m->getMechTo()->getContext();

//         std::cerr << "push an Edge Set Mechanical !!!!!!!!!!! :"  << from->getName() << " and " << to->getName() << " with " << m->getName() << "\n";
    edgeToAdd.insert(std::make_pair(from, to));
//         addHedge( h_node_vertex_map[from], h_node_vertex_map[to] );
// 	addRedge( r_node_vertex_map[to], r_node_vertex_map[from] );
}
/// Method called when a MechanicalMapping is destroyed.
void BglSimulation::resetMechanicalMapping(Node* , core::componentmodel::behavior::BaseMechanicalMapping* m )
{

    Node *from=(Node*)m->getMechFrom()->getContext();
    Node *to=(Node*)m->getMechTo()->getContext();
//         std::cerr << "Reset Mechanical !!!!!!!!!!! :"  << from->getName() << " and " << to->getName() << " with " << m->getName() << "\n";
    removeHedge( h_node_vertex_map[from], h_node_vertex_map[to] );
    removeRedge( r_node_vertex_map[to], r_node_vertex_map[from] );
}

/// Method called when a MechanicalMapping is created.
void BglSimulation::setContactResponse(Node * parent, core::objectmodel::BaseObject* response)
{
    if (InteractionForceField *iff = dynamic_cast<InteractionForceField*>(response))
    {
//             std::cerr << "ADD CONTACT RESPONSE " << parent->getName() << "   : " << response->getName() << "\n";
        addInteraction( (Node*)iff->getMechModel1()->getContext(),
                (Node*)iff->getMechModel2()->getContext(),
                iff);
    }
    else if (// InteractionConstraint *ic =
        dynamic_cast<InteractionConstraint*>(response))
    {
        std::cerr << "setContactResponse Not IMPLEMENTED YET for InteractionConstraint\n";
    }
}
/// Method called when a MechanicalMapping is destroyed.
void BglSimulation::resetContactResponse(Node * parent, core::objectmodel::BaseObject* response)
{
    if (InteractionForceField *iff = dynamic_cast<InteractionForceField*>(response))
    {

//             std::cerr << "ReSet ContactResponse : " << iff->getMechModel1()->getContext()->getName() << "->"<< iff->getMechModel2()->getName() << " : "
//                       << iff->getMechModel2()->getContext()->getName() << "->" <<iff->getMechModel2()->getName() << "\n";
        removeInteraction(iff);
    }
    else if (// InteractionConstraint *ic =
        dynamic_cast<InteractionConstraint*>(response))
    {
        std::cerr << "setContactResponse Not IMPLEMENTED YET for InteractionConstraint\n";
    }
}


void BglSimulation::addInteraction( Node* n1, Node* n2, InteractionForceField* iff )
{
    interactionToAdd.push_back( InteractionData(n1, n2, iff) );
}

void BglSimulation::addInteractionNow( InteractionData &i )
{
    interactions.push_back( Interaction(h_node_vertex_map[i.n1], h_node_vertex_map[i.n2],i.iff));
    needUpdateInteraction=true;
}

void BglSimulation::removeInteraction( InteractionForceField* iff )
{
    Interactions::iterator it;
    for (it=interactions.begin(); it!=interactions.end(); ++it)
    {
        if (it->iff == iff)
        {
//                 ((Node*)(iff->getContext()))->removeObject(iff);
            interactions.erase(it);
            needUpdateInteraction=true;
            return;
        }
    }
    InteractionsData::iterator itData;
    for (itData=interactionToAdd.begin(); itData!=interactionToAdd.end(); itData++)
    {
        if (itData->iff == iff)
        {
            interactionToAdd.erase(itData);
            return;
        }
    }
//         std::cerr << iff << "@" << iff->getName() << " : ########################################## NO REMOVAL\n";
}

namespace
{

class find_leaves: public ::boost::bfs_visitor<>
{
public:
    typedef vector<BglSimulation::Rvertex> Rleaves;
    Rleaves& leaves; // use external data, since internal data seems corrupted after the visit (???)
    find_leaves( Rleaves& l ):leaves(l) {}
    void discover_vertex( BglSimulation::Rvertex v, const BglSimulation::Rgraph& g )
    {
        if ( out_degree (v,g)==0 )  // leaf vertex
        {
            leaves.push_back(v);
        }
    }

};
}

/**
Data: hroots, interactions
 Result: interactionGroups
    */
void BglSimulation::computeInteractionGraphAndConnectedComponents()
{
    i_vertex_node_map = get( bglnode_t(), igraph);
    i_edge_interaction_map = get( interaction_t(), igraph );

    // create the interaction graph vertices (root nodes only)
    igraph.clear();
    for ( HvertexVector::iterator i=hroots.begin(), iend=hroots.end(); i!=iend; i++ )
    {

        Ivertex iv = add_vertex( igraph );
        Node* n = h_vertex_node_map[*i];
//             std::cerr << "hroots: " << n->getName() << "\n";
        i_vertex_node_map[iv] = n;
        i_node_vertex_map[n] = iv;
    }

//         cerr<<"interaction nodes: "<<endl;
//         for( Ivpair i = vertices(igraph); i.first!=i.second; i.first++ )
//           cerr<<i_vertex_node_map[*i.first]->getName()<<", ";
//         cerr<<endl;
//         cerr<<"begin create interaction edges"<<endl;

    // create the edges between the root nodes and associate the interactions with the root nodes
    // rgraph is used to find the roots corresponding to the nodes.
    typedef std::map<Rvertex,Interactions > R_vertex_interactions_map;
    R_vertex_interactions_map rootInteractions;
    for ( Interactions::iterator i=interactions.begin(), iend=interactions.end(); i!=iend; i++ )
    {
//               cerr<<"find all the roots associated with the interaction from "<<h_vertex_node_map[(*i).v1]->getName()<<" to "<<h_vertex_node_map[(*i).v2]->getName()<<endl;

        // find all the roots associated with the given interaction
        vector<BglSimulation::Rvertex> leaves;
        find_leaves visit(leaves);
        boost::vector_property_map<boost::default_color_type> colors( boost::num_vertices(rgraph) );
        boost::queue<Rvertex> queue;
        Rvertex v1 = r_node_vertex_map[ h_vertex_node_map[ (*i).v1 ]];
        Rvertex v2 = r_node_vertex_map[ h_vertex_node_map[ (*i).v2 ]];
        boost::breadth_first_search( rgraph, boost::vertex(v1, rgraph), queue, visit, colors );
        boost::breadth_first_search( rgraph, boost::vertex(v2, rgraph), queue, visit, colors );

//             cerr<<"the roots are: "<<endl;
//             for( unsigned j=0; j<leaves.size(); j++ ){
//               cerr<<r_vertex_node_map[visit.leaves[j]]->getName()<<", ";
//             }
//             cerr<<endl;

        // associate the interaction with one of its roots, no matter which one because it will then be associated to the whole interaction group.
        assert( visit.leaves.size()>0 );
        rootInteractions[*visit.leaves.begin()].push_back( *i );

//             Rvertex collisionRvertex = convertHvertex2Rvertex(collisionVertex);
        // add edges between all the pairs of roots
        for ( find_leaves::Rleaves::iterator l=visit.leaves.begin(), lend=visit.leaves.end(); l!=lend; l++ )
        {
//                 if (*l == collisionRvertex) continue;
            for ( find_leaves::Rleaves::iterator m=l++; m!=lend; m++ )
            {
//                     if (*m == collisionRvertex) continue
                ;
                if ( *l != *m )
                {
                    std::pair<Iedge,bool> e = add_edge( i_node_vertex_map[r_vertex_node_map[*l]],i_node_vertex_map[r_vertex_node_map[*m]], igraph );
                    assert( e.second );
                }
            }
        }
    }

    // compute the connected components of the interaction graph, represented by integers associated with vertices
    vector<int> component(num_vertices(igraph));
    int num = boost::connected_components(igraph, &component[0]);

    // build the interactionGroups
    interactionGroups.clear();
    interactionGroups.resize(num);
    int index = 0;
    for ( Ivpair i=vertices(igraph); i.first!=i.second; i.first++,index++ )
    {
        Ivertex iv = *i.first;
        Hvertex hv = h_node_vertex_map[ i_vertex_node_map[iv]]; // hv(iv)
        Rvertex rv = r_node_vertex_map[ i_vertex_node_map[iv]]; // rv(iv)
        InteractionGroup& group = interactionGroups[component[index]]; // the group the node belongs to

        group.first.push_back( hv );    // add the node to its interaction group
        // add its associated interaction to the interaction group
        for ( Interactions::iterator j=rootInteractions[rv].begin(), jend=rootInteractions[rv].end(); j!=jend; j++ )
        {
            group.second.push_back( *j );
        }
    }
    // 	debug
    // 	     cerr<<"end connected components"<<endl;
//         for( unsigned i=0; i<interactionGroups.size(); i++ )
//           {
//             cerr<<"interaction group (roots only): "<<endl;
//             cerr<<"- nodes = ";
//             for( unsigned j=0; j<interactionGroups[i].first.size(); j++ )
//               {
//                 Node* root = h_vertex_node_map[ interactionGroups[i].first[j] ];
//                 cerr<< root->getName() <<", ";
//               }
//             //             cerr<<endl<<"- interactions = ";
//             //             for( unsigned j=0; j<interactionGroups[i].second.size(); j++ )
//             //               {
//             //                 Node* n1 = r_vertex_node_map[ interactionGroups[i].second[j].v1 ];
//             //                 Node* n2 = r_vertex_node_map[ interactionGroups[i].second[j].v2 ];
//             //                 InteractionForceField* iff = interactionGroups[i].second[j].iff;
//             //                 cerr<<iff->getName()<<" between "<<n1->getName()<<" and "<<n2->getName()<< ", ";
//             //               }
//             cerr<<endl;
//           }
    needUpdateInteraction=false;
}

void BglSimulation::computeCollisionGraph()
{

    for ( Hvpair iter=boost::vertices(hgraph); iter.first!=iter.second; iter.first++)
    {
        if (h_vertex_node_map[*iter.first]->collisionModel.size())
        {
//                 cerr<<"Node "<<h_vertex_node_map[*iter.first]->getName()<<" is a collision Node"<<endl;
            addHedge(collisionVertex,*iter.first );
        }
    }
}

void BglSimulation::setSolverOfCollisionGroup(Node* solverNode, Node* solverOfCollisionGroup)
{
    solver_colisionGroup_map[solverNode] = solverOfCollisionGroup;
}



bool BglSimulation::isUsableRoot(Node* n)
{
    if (n==masterNode || n == collisionNode) return false;
    std::map< Node*, Node* >::const_iterator itSolver;
    for (itSolver=node_solver_map.begin(); itSolver!=node_solver_map.end(); itSolver++)
    {
        if (itSolver->second == n) return false;
    }
    return true;
}

/**
Data: hgraph, rgraph
 Result: hroots, interaction groups, all nodes initialized.
    */
void BglSimulation::init()
{
    cerr<<"begin BglSimulation::init()"<<endl;

    /// find the roots in hgraph
    computeHroots();

    /// Initialize all the nodes using a depth-first visit
    for (HvertexVector::iterator i=hroots.begin(), iend=hroots.end(); i!=iend; i++ )
    {
        h_vertex_node_map[*i]->init(); // init the root and its subgraph
        //Link all the nodes to the MasterNode, in ordre to propagate the Context
        addEdge( masterVertex, *i); // add nodes
    }

    masterNode->init();

    /// compute the interaction groups
    computeInteractionGraphAndConnectedComponents();

    computeCollisionGraph();

    /// initialize the visual models and their mappings
    for ( VisualVector::iterator i=visualModels.begin(), iend=visualModels.end(); i!=iend; i++ )
    {
        (*i)->init();
    }
    for ( MappingVector::iterator i=visualMappings.begin(), iend=visualMappings.end(); i!=iend; i++ )
    {
        (*i)->init();
    }
}

/** Data: interaction groups
Result: nodes updated
*/

void BglSimulation::mechanicalStep(Node* root, double dt)
{
    /** Put everyone in the graph to apply collision detection */
//         cerr<<"BglSimulation::animate, start collision detection"<<endl;
//         print(masterNode);
    clearVertex( masterVertex );
//         masterNode->clearInteractionForceFields();

    if (collisionPipeline)
    {
        masterNode->moveObject( collisionPipeline );
        addEdge(masterVertex,collisionVertex);


//             std::cerr << "\n\nStatus before Collision Detection:\n";
//             print(masterNode);

        CollisionVisitor act;
        masterNode->doExecuteVisitor(&act);

        removeEdge( masterVertex, collisionVertex);
        masterNode->removeObject( collisionPipeline );
    }

    updateGraph();

//         std::cerr << "\n\nStatus After Update:\n";
//         print(masterNode);

    for ( unsigned i=0; i<interactionGroups.size(); i++ )
    {
        // remove previous children and interactions

        clearVertex( masterVertex );

        // add the vertices and the interactions
        InteractionGroup& group = interactionGroups[i];

        //Find Dynamic Objects: objects with solver
        std::multimap< Node*, Hvertex>  solverNode_dynamicObject_map;
        std::set< Hvertex > staticObjectAdded;
        std::set< Hvertex > solverUsed;
        //Find Static  Objects: objects without solver

        std::string staticObjectName;
        for ( unsigned j=0; j<group.first.size(); j++ )
        {
            Hvertex currentVertex = group.first[j];
            Node   *currentNode   = h_vertex_node_map[ currentVertex ];
//                 std::cerr << "currentNode : " << currentNode->getName() << "\n";


            //No solver Found: we link it to the masterNode
            if (node_solver_map.find( currentNode ) == node_solver_map.end())
            {
                addEdge( masterVertex, currentVertex); // add static object
                staticObjectAdded.insert(currentVertex);
                staticObjectName += currentNode->getName() + " ";
            }
            else
            {
                Node *sUsed=node_solver_map[ currentNode ];
                //Verify if the current solver is not controled by a collision group
                if (solver_colisionGroup_map.find(sUsed) != solver_colisionGroup_map.end())
                    sUsed = solver_colisionGroup_map[sUsed];

                //Add the main solver, and link it to the vertex describing the dynamic object
                solverNode_dynamicObject_map.insert(std::make_pair(sUsed, currentVertex) );
                solverUsed.insert(h_node_vertex_map[sUsed]);
            }
        }


        //We deal with all the solvers one by one
        std::set< Hvertex >::iterator it;
        for (it=solverUsed.begin(); it != solverUsed.end(); it++)
        {
            std::string animationName;
            Hvertex solverVertex =*it;
            Node* currentSolver=h_vertex_node_map[solverVertex];
//                 std::cerr << "Solving with " << (*it)->getName() << "\n";

            typedef std::multimap< Node*,Hvertex>::iterator dynamicObjectIterator;
            std::pair<dynamicObjectIterator,dynamicObjectIterator> rangeSolver;
            rangeSolver = solverNode_dynamicObject_map.equal_range( currentSolver );


            if (rangeSolver.first == rangeSolver.second) continue; //No dynamic object, so no need to animate

            //Link the solver to the master Node
            addEdge( masterVertex,solverVertex);
            //Add all the dynamic object depending of the current solver
            for (dynamicObjectIterator itDynObj=rangeSolver.first; itDynObj!=rangeSolver.second; itDynObj++)
            {
//                     std::cerr << "\tAdding " << h_vertex_node_map[itDynObj->second]->getName() << "\n";
                animationName += h_vertex_node_map[itDynObj->second]->getName() + " " ;
                addEdge( solverVertex,itDynObj->second ); // add nodes
            }


//                 std::cerr << "\n\nStatus before Animating:\n";
//                 print(masterNode);
#ifdef DUMP_VISITOR_INFO
            simulation::Visitor::printComment(std::string("Animate ") + staticObjectName + animationName );
#endif
            // animate this interaction group
            masterNode->animate(dt);

            //Remove the link between all the vertices connected to the solver
            removeEdge( masterVertex, solverVertex);
            //Clear the solverVertex from all the dynamic objects
            clearVertex(solverVertex);
        }

        for (std::set<Hvertex>::iterator it=staticObjectAdded.begin(); it!=staticObjectAdded.end(); it++) removeEdge(masterVertex,*it);

//             std::cerr << "\n\nFinal Status:\n\n\n";
//             clear_vertex(masterVertex, hgraph);
//             print(masterNode);
    }

    //TODO: add the interactions to the graph
    for ( HvertexVector::iterator i=hroots.begin(), iend=hroots.end(); i!=iend; i++ )
    {
//             std::cerr << h_vertex_node_map[*i]->getName() << " : root !! \n";
        addEdge( masterVertex,*i);
    }
}



/// TODO: adapt the AnimateVisitor to BGL
void BglSimulation::animate(Node* root, double dt)
{
    dt = root->getContext()->getDt();

    clearVertex( masterVertex );

#ifdef DUMP_VISITOR_INFO
    simulation::Visitor::printComment(std::string("Begin Step"));
#endif
    {
        AnimateBeginEvent ev ( dt );
        PropagateEventVisitor act ( &ev );
        masterNode->doExecuteVisitor ( &act );
    }


    double startTime = root->getTime();
    double mechanicalDt = dt/numMechSteps.getValue();

    BehaviorUpdatePositionVisitor beh(dt);

    for( unsigned step=0; step<numMechSteps.getValue(); step++ )
    {
        mechanicalStep(root,dt);
        masterNode->doExecuteVisitor ( &beh );
        masterNode->setTime ( startTime + (step+1)* mechanicalDt );
    }

    //Update Visual Mapping

    for ( MappingVector::iterator i=visualMappings.begin(), iend=visualMappings.end(); i!=iend; i++ )
    {
        //cerr<<"init mapping model, mapped from "<<(*i).second->getFrom()->getName()<<" to "<<(*i).second->getTo()->getName()<<endl;
        (*i)->updateMapping();
    }
    //Update Mapping End Event
    {
        simulation::UpdateMappingEndEvent ev ( dt );
        PropagateEventVisitor act ( &ev );
        masterNode->doExecuteVisitor( &act );
    }

    {
        AnimateEndEvent ev ( dt );
        PropagateEventVisitor act ( &ev );
        masterNode->doExecuteVisitor( &act );
    }
#ifdef DUMP_VISITOR_INFO
    simulation::Visitor::printComment(std::string("End Step"));
#endif

}

void BglSimulation::computeHroots()
{
    /// find the roots in hgraph
    hroots.clear();
    for ( Hvpair iter=boost::vertices(hgraph); iter.first!=iter.second; iter.first++)
    {
//                 std::cerr << isUsableRoot(h_vertex_node_map[*iter.first]) << "-->usable? : " << h_vertex_node_map[*iter.first]->getName() << " " << in_degree (*iter.first,hgraph) << "\n";
        if ( isUsableRoot(h_vertex_node_map[*iter.first]) && in_degree (*iter.first,hgraph)==0)
        {
            hroots.push_back(*iter.first);
            //                   cerr<<"node "<<h_vertex_node_map[*iter.first]->getName()<<" is a root"<<endl;
        }
    }
}

// after the collision detection, modification to the graph can have occured. We update the graph
void BglSimulation::updateGraph()
{
//         std::cerr << "node:" <<nodeToAdd.size() << " : edge:" << edgeToAdd.size() << " : interaction:" << interactionToAdd.size() << "\n";
    for (std::set<Hvertex>::iterator it=vertexToDelete.begin(); it!=vertexToDelete.end(); it++) deleteVertex(*it);
    for (unsigned int i=0; i<nodeToAdd.size() ; i++)               insertNewNode(nodeToAdd[i]);
    std::set< std::pair< Node*, Node*> >::iterator itEdge;
    for (itEdge=edgeToAdd.begin(); itEdge!=edgeToAdd.end(); itEdge++) addEdge(h_node_vertex_map[ itEdge->first],h_node_vertex_map[ itEdge->second]);
    for (unsigned int i=0; i<interactionToAdd.size(); i++)         addInteractionNow(interactionToAdd[i]);

    if (vertexToDelete.size() || nodeToAdd.size()) computeHroots();

    vertexToDelete.clear();
    nodeToAdd.clear();
    edgeToAdd.clear();
    interactionToAdd.clear();

    if (needUpdateInteraction)   computeInteractionGraphAndConnectedComponents();

}


void BglSimulation::computeBBox(Node* root, SReal* minBBox, SReal* maxBBox)
{
    sofa::simulation::Simulation::computeBBox(root,minBBox,maxBBox);

    for ( VisualVector::iterator i=visualModels.begin(), iend=visualModels.end(); i!=iend; i++ )
    {
        (*i)->addBBox(minBBox,maxBBox);
    }
}

void BglSimulation::setVisualModel( VisualModel* v)
{
    visualModels.push_back( v );
}
void BglSimulation::setVisualMapping( Mapping* m)
{
    visualMappings.push_back( m );
}

void BglSimulation::draw(Node* , helper::gl::VisualParameters*)
{
    if (!masterNode) return;
    // 	cerr<<"begin BglSimulation::glDraw()"<<endl;
    masterNode->glDraw();


    for ( VisualVector::iterator i=visualModels.begin(), iend=visualModels.end(); i!=iend; i++ )
    {
        (*i)->drawVisual();
    }
    for ( MappingVector::iterator i=visualMappings.begin(), iend=visualMappings.end(); i!=iend; i++ )
    {
        (*i)->draw();
    }

    // 	cerr<<"end BglSimulation::glDraw()"<<endl;
}


/// Create a GNode tree structure using available file loaders, then convert it to a BglSimulation
Node* BglSimulation::load(const char* f)
{
    std::string fileName(f);
    /*  if (fileName.empty())*/
    {
        //         fileName = "liver.scn";
        sofa::helper::system::DataRepository.findFile(fileName);
    }

    std::cerr << "Loading " << f << "\n";

    sofa::simulation::tree::GNode* groot = 0;

    sofa::simulation::tree::TreeSimulation treeSimu;
    std::string in_filename(fileName);
    if (in_filename.rfind(".simu") == std::string::npos)
        groot = dynamic_cast< sofa::simulation::tree::GNode* >(treeSimu.load(fileName.c_str()));

    std::cerr << "File Loaded\n";
    if ( !groot )
    {
        cerr<<"BglSimulation::load file "<<fileName<<" failed"<<endl;
        exit(1);
    }
    //else cerr<<"BglSimulation::loaded file "<<fileName<<endl;

    //      cerr<<"GNode loaded: "<<endl;
    //      groot->execute<PrintVisitor>();
    //      cerr<<"==========================="<<endl;
    std::map<simulation::Node*,BglNode*> gnode_bnode_map;
    BuildNodesFromGNodeVisitor b1(this);
    groot->execute(b1);
    gnode_bnode_map = b1.getGNodeBNodeMap();
    BuildRestFromGNodeVisitor b2(this);
    b2.setGNodeBNodeMap(gnode_bnode_map);
    groot->execute(b2);


    const sofa::core::objectmodel::Context &c = *( (sofa::core::objectmodel::Context*)groot->getContext());
    masterNode->copyContext(c);

    init();

    return masterNode;

    /*    cerr<<"loaded graph has "<<num_vertices(hgraph)<<" vertices and "<<num_edges(hgraph)<<" edges:"<<endl;
          PrintVisitor printvisitor;
          dfs( printvisitor );*/
}
/// Add an OdeSolver working inside a given Node
void BglSimulation::addSolver(BaseObject* s,Node* n)
{
    if (!n) masterNode->moveObject(s);
    else
    {
        Node *solverNode = node_solver_map[n];
        if (!solverNode) //Node not created yet
        {
            std::ostringstream s;
            s << "SolverNode" << node_solver_map.size();
            solverNode = newNodeNow(s.str());
            node_solver_map[n]=solverNode;
            //                 addHedge( masterVertex, h_node_vertex_map[solverNode]);
            //                 addRedge( r_node_vertex_map[solverNode],masterVertex);
        }
//             std::cerr << "Adding Solver : " << s->getName() << " To " << solverNode->getName() << "\n";
        solverNode->moveObject(s);
    }
}


void BglSimulation::reset(Node* root)
{
    for ( unsigned i=0; i<interactionGroups.size(); i++ )
    {
        interactionGroups[i].second.clear();
    }
    solver_colisionGroup_map.clear();
    sofa::simulation::Simulation::reset(root);
}

void BglSimulation::unload(Node* root)
{

    if (!root) return;

    for ( unsigned i=0; i<interactionGroups.size(); i++ )
    {
        interactionGroups[i].second.clear();
    }
    solver_colisionGroup_map.clear();

    instruments.clear();
    instrumentInUse.setValue(-1);
    root->execute<CleanupVisitor>();
    BglDeleteVisitor deleteGraph(this);
    masterNode->doExecuteVisitor(&deleteGraph);



    for ( VisualVector::iterator i=visualModels.begin(), iend=visualModels.end(); i!=iend; i++ )
    {
        delete (*i);
    }
    for ( MappingVector::iterator i=visualMappings.begin(), iend=visualMappings.end(); i!=iend; i++ )
    {
        delete (*i);
    }
    visualModels.clear();
    visualMappings.clear();


    vertexToDelete.clear();
    nodeToAdd.clear();
    edgeToAdd.clear();
    interactionToAdd.clear();


    hgraph.clear();
    h_node_vertex_map.clear();
    rgraph.clear();
    r_node_vertex_map.clear();
    delete collisionPipeline;
    collisionPipeline=NULL;

    delete masterNode; masterNode=NULL;
    delete collisionNode; collisionNode=NULL;

    h_vertex_node_map = get( bglnode_t(), hgraph);
    r_vertex_node_map = get( bglnode_t(), rgraph);

    // The animation control overloads the solvers of the scene
    masterNode= static_cast<BglNode*>(newNodeNow("masterNode"));
    masterVertex = h_node_vertex_map[masterNode];

    // The collision Node
    collisionNode= static_cast<BglNode*>(newNodeNow("collisionNode"));
    collisionVertex = h_node_vertex_map[collisionNode];


}

/// depth search in the whole scene
void BglSimulation::dfs( Visitor& visit )
{
    dfs_adapter vis(&visit,h_vertex_node_map);
    boost::depth_first_search( hgraph, boost::visitor(vis) );
}
/// depth search starting from the given vertex, and prunable
void BglSimulation::dfv( Hvertex v, Visitor& vis )
{
    boost::vector_property_map<boost::default_color_type> colors( boost::num_vertices(hgraph) );
    dfv_adapter dfsv(&vis,h_vertex_node_map );
    boost::depth_first_visit(
        hgraph,
        boost::vertex(v, hgraph),
        dfsv,
        colors,
        dfsv
    );
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


