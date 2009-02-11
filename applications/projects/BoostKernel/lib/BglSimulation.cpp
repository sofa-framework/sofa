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

BglSimulation::Hedge BglSimulation::addHedge( Hvertex p, Hvertex c )
{
    std::pair<Hedge, bool> e =  add_edge(  p,c,hgraph);
    assert(e.second);
    return e.first;
}

BglSimulation::Redge BglSimulation::addRedge( Rvertex p, Rvertex c )
{
    std::pair<Redge, bool> e =  add_edge(  p,c,rgraph);
    assert(e.second);
    return e.first;
}

BglSimulation::Hedge BglSimulation::addCedge( Hvertex p, Hvertex c )
{
    std::pair<Hedge, bool> e =  add_edge(  p,c,hgraph);
    assert(e.second);
    return e.first;
}

BglSimulation::Hvertex BglSimulation::addCvertex(Node *n)
{
    Hvertex v = add_vertex( hgraph);
    h_vertex_node_map[v] = n;
    h_node_vertex_map[n] = v;

    addCedge(collisionVertex, v);
    return v;
}

void BglSimulation::removeHedge( Hvertex p, Hvertex c )
{
    remove_edge(  p,c,hgraph);
}

void BglSimulation::removeRedge( Rvertex p, Rvertex c )
{
    remove_edge(  p,c,rgraph);
}

void BglSimulation::removeCedge( Hvertex p, Hvertex c )
{
    remove_edge(  p,c,hgraph);
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
    h_vertex_node_map[ hnode ] = bglN;
    h_node_vertex_map[ bglN ] = hnode;

    //Reverse graph
    Rvertex rnode = add_vertex( rgraph );
    r_vertex_node_map[rnode] = bglN;
    r_node_vertex_map[bglN] = rnode;

    //configure the node
    bglN->graph = &hgraph;
    bglN->vertexId = hnode;
}


Node* BglSimulation::newNodeNow(const std::string& name)
{
//         std::cerr << "new Node : " << name << "\n";
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
    if (parent != masterNode)
        edgeToAdd.push_back(std::make_pair(parent,child));
}

/// Add a node to as the child of another
void BglSimulation::addNodeNow(BglNode* parent, BglNode* child)
{
//           std::cerr << "ADD Node NOW " << parent->getName() << "-->" << child->getName() << " : \n";
    addHedge( h_node_vertex_map[parent], h_node_vertex_map[child] );
    addRedge( r_node_vertex_map[child], r_node_vertex_map[parent] );
}


void BglSimulation::deleteNode( Node* n)
{
//         std::cerr << "Delete Node " << n << " : \n";
    nodeToDelete.insert(n);
}

void BglSimulation::deleteNodeNow( Node* n)
{
//         std::cerr << "Delete Now Node " << n << " : \n";
//         std::cerr << n->getName() << " : " << h_node_vertex_map[n] << "\n";
    Hvertex vH=h_node_vertex_map[n];
    clear_vertex(vH, hgraph);
    //         remove_vertex(vH, hgraph);
    h_node_vertex_map.erase(n);

    Rvertex vR=r_node_vertex_map[n];
    clear_vertex(vR, rgraph);
    //         remove_vertex(vR, rgraph);
    r_node_vertex_map.erase(n);
}



/// Method called when a MechanicalMapping is created.
void BglSimulation::setMechanicalMapping(Node* c, core::componentmodel::behavior::BaseMechanicalMapping* m )
{
    Node *from=(Node*)m->getMechFrom()->getContext();
    Node *to=(Node*)m->getMechTo()->getContext();
//         std::cerr << "Set Mechanical !!!!!!!!!!! :"  << from->getName() << " and " << to->getName() << " with " << m->getName() << "\n";
    addHedge( h_node_vertex_map[from], h_node_vertex_map[to] );
    addRedge( r_node_vertex_map[to], r_node_vertex_map[from] );
    c->moveObject(m);
}
/// Method called when a MechanicalMapping is destroyed.
void BglSimulation::resetMechanicalMapping(Node* c, core::componentmodel::behavior::BaseMechanicalMapping* m )
{

    Node *from=(Node*)m->getMechFrom()->getContext();
    Node *to=(Node*)m->getMechTo()->getContext();
//         std::cerr << "Reset Mechanical !!!!!!!!!!! :"  << from->getName() << " and " << to->getName() << " with " << m->getName() << "\n";
    removeHedge( h_node_vertex_map[from], h_node_vertex_map[to] );
    removeRedge( r_node_vertex_map[to], r_node_vertex_map[from] );
    c->removeObject(m);
}

/// Method called when a MechanicalMapping is created.
void BglSimulation::setContactResponse(Node * parent, core::objectmodel::BaseObject* response)
{
    std::cerr << "Set Contact\n";
    if (InteractionForceField *iff = dynamic_cast<InteractionForceField*>(response))
    {
        std::cerr << "ADD CONTACT RESPONSE " << parent->getName() << "   : " << response->getName() << "\n";
        addInteraction( (Node*)iff->getMechModel1()->getContext(),
                (Node*)iff->getMechModel2()->getContext(),
                iff);
    }
    else if (// InteractionConstraint *ic =
        dynamic_cast<InteractionConstraint*>(response))
    {
        std::cerr << "setContactResponse Not IMPLEMENTED YET for InteractionConstraint\n";
    }
    else
    {
        parent->moveObject(response);
        //               std::cerr << "Add Object " << response->getName() << "\n";
    }
}
/// Method called when a MechanicalMapping is destroyed.
void BglSimulation::resetContactResponse(Node * parent, core::objectmodel::BaseObject* response)
{
    if (InteractionForceField *iff = dynamic_cast<InteractionForceField*>(response))
    {

        //               std::cerr << "ReSet ContactResponse : " << iff->getMechModel1()->getContext()->getName() << "->"<< iff->getMechModel1()->getName() << " : "
        //                         << iff->getMechModel2()->getContext()->getName() << "->" <<iff->getMechModel2()->getName() << "\n";
        removeInteraction(iff);
    }
    else if (// InteractionConstraint *ic =
        dynamic_cast<InteractionConstraint*>(response))
    {
        std::cerr << "setContactResponse Not IMPLEMENTED YET for InteractionConstraint\n";
    }
    else
    {
        //               std::cerr << "Remove Object " << response->getName() << " from " << parent->getName() << " and " << response->getContext()->getName() << " " << dynamic_cast<BglNode*>(parent) << "\n";
        parent->removeObject(response);
    }
}


void BglSimulation::addInteraction( Node* n1, Node* n2, InteractionForceField* iff )
{
    interactionToAdd.push_back( InteractionData(n1, n2, iff) );
}

void BglSimulation::addInteractionNow( InteractionData &i )
{
    interactions.push_back( Interaction(h_node_vertex_map[i.n1], h_node_vertex_map[i.n2],i.iff));
}

void BglSimulation::removeInteraction( InteractionForceField* iff )
{
    Interactions::iterator it;
    for (it=interactions.begin(); it!=interactions.end(); ++it)
    {
        if (it->iff == iff)
        {
            ((Node*)(iff->getContext()))->removeObject(iff);
            interactions.erase(it); return;
        }
    }
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

        cerr<<"the roots are: "<<endl;
        for( unsigned j=0; j<leaves.size(); j++ )
        {
            cerr<<r_vertex_node_map[visit.leaves[j]]->getName()<<", ";
        }
        cerr<<endl;

        // associate the interaction with one of its roots, no matter which one because it will then be associated to the whole interaction group.
        assert( visit.leaves.size()>0 );
        rootInteractions[*visit.leaves.begin()].push_back( *i );

        // add edges between all the pairs of roots
        for ( find_leaves::Rleaves::iterator l=visit.leaves.begin(), lend=visit.leaves.end(); l!=lend; l++ )
        {
            for ( find_leaves::Rleaves::iterator m=l++; m!=lend; m++ )
            {
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

}

void BglSimulation::computeCollisionGraph()
{

    for ( Hvpair iter=boost::vertices(hgraph); iter.first!=iter.second; iter.first++)
    {
        if (h_vertex_node_map[*iter.first]->collisionModel.size())
        {
//                 cerr<<"Node "<<h_vertex_node_map[*iter.first]->getName()<<" is a collision Node"<<endl;
            addCvertex(h_vertex_node_map[*iter.first]);
        }
    }
}

void BglSimulation::setSolverOfCollisionGroup(Node* solverNode, Node* solverOfCollisionGroup)
{
    solver_colisionGroup_map[solverNode] = solverOfCollisionGroup;
}

bool BglSimulation::isIFFinNode(Rvertex IFFNode, Rvertex n)
{
    Rgraph::out_edge_iterator out_i, out_end;
//         std::cerr << "processing : " << r_vertex_node_map[IFFNode]->getName()
//                   << ":" << r_vertex_node_map[n]->getName() << "\n";
    if (IFFNode == n) return true;
    for (tie(out_i, out_end) = out_edges(IFFNode, rgraph); out_i!=out_end; ++out_i)
    {
        Redge e=*out_i;
        Rvertex v= target(e,rgraph);
        if (isIFFinNode(v,n)) return true;
    }
    return false;
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
        addHedge( masterVertex, *i); // add nodes
    }

    masterNode->init(); // not in hroots
    /// compute the interaction groups
    computeInteractionGraphAndConnectedComponents();

    computeCollisionGraph();

    /// initialize the visual models and their mappings
    for ( VisualVector::iterator i=visualModels.begin(), iend=visualModels.end(); i!=iend; i++ )
    {
        //cerr<<"init visual model, mapped from "<<(*i).second->getFrom()->getName()<<" to "<<(*i).second->getTo()->getName()<<endl;
        (*i).first->init();
        (*i).second->init();
    }




    /// Put the collision models in the collision graph
    /*      for( Hvpair v = vertices(hgraph); v.first!=v.second; v.first++ )
    	{
                BglNode* node = h_vertex_node_map[*v.first];
                if( !node->collisionModel.empty() )
                {
    	// insert the node in the collision graph
    	Hvertex cvertex = add_vertex( cgraph );
    	c_vertex_node_map[cvertex] = node;
    	cerr<<"add "<<node->getName()<<" to the collision graph"<<endl;
                }
    	}*/



    //cerr<<"end BglSimulation::init()"<<endl;

    // debug
    //      cerr<<"there are "<<num_vertices(hgraph)<<" hvertices:"<<endl;
    //      for( Hvpair i=vertices(hgraph); i.first!=i.second; i.first++ )
    //      {
    //              cerr<<h_vertex_node_map[*i.first]->getName()<<", ";
    //      }
    //      cerr<<endl;
    //      cerr<<"there are "<<num_vertices(rgraph)<<" rvertices:"<<endl;
    //      for( Rvpair i=vertices(rgraph); i.first!=i.second; i.first++ )
    //      {
    //              cerr<<r_vertex_node_map[*i.first]->getName()<<", ";
    //      }
    //      cerr<<endl;

}

/** Data: interaction groups
Result: nodes updated
*/

void BglSimulation::mechanicalStep(Node* root, double dt)
{
    /** Put everyone in the graph to apply collision detection */
//         cerr<<"BglSimulation::animate, start collision detection"<<endl;
//           print(masterNode);
    clear_vertex( masterVertex, hgraph );
    masterNode->clearInteractionForceFields();

    if (collisionPipeline)
    {
        masterNode->moveObject( collisionPipeline );
        addHedge(masterVertex,collisionVertex);


//             std::cerr << "\n\nStatus before Collision Detection:\n";
//             print(masterNode);

        CollisionVisitor act;
        masterNode->doExecuteVisitor(&act);

        removeHedge( masterVertex, collisionVertex);
        masterNode->removeObject( collisionPipeline );
    }

    updateGraph();

//                   std::cerr << "\n\nStatus After Update:\n";
//                   print(masterNode);

    /** Update each interaction group independently.
        The master node is used to process all of them sequentially, but several copies of the master node would allow paralle processing.
    */
//          std::cerr << interactionGroups.size() << " interactions \n";

    if (interactions.size())
        computeInteractionGraphAndConnectedComponents();


//         std::cerr << "\n\n\n\n";
    for ( unsigned i=0; i<interactionGroups.size(); i++ )
    {
//             std::cerr << "\n\nInteraction " << i << "\n";
        // remove previous children and interactions

        clear_vertex( masterVertex, hgraph );
        masterNode->clearInteractionForceFields();

        // add the vertices and the interactions
        InteractionGroup& group = interactionGroups[i];

        //Find Dynamic Objects: objects with solver
        std::multimap< Node*, Hvertex>                solverNode_dynamicObject_map;
        std::set< Node* > solverUsed;
        std::vector< Hedge > staticObjectAdded;
        //Find Static  Objects: objects without solver


        for ( unsigned j=0; j<group.first.size(); j++ )
        {
            Hvertex currentVertex = group.first[j];
            Node   *currentNode   = h_vertex_node_map[ currentVertex ];
//                 std::cerr << "currentNode : " << currentNode->getName() << "\n";


            //No solver Found: we link it to the masterNode
            if (node_solver_map.find( currentNode ) == node_solver_map.end())
            {
//                     std::cerr <<"#############################Static Object: " << currentNode->getName() << "\n";
                BglSimulation::Hedge eH=addHedge( masterVertex, currentVertex); // add static object
                staticObjectAdded.push_back(eH);
            }
            else
            {
                Node *sUsed=node_solver_map[ currentNode ];
                //Verify if the current solver is not controled by a collision group
                if (solver_colisionGroup_map.find(sUsed) != solver_colisionGroup_map.end())
                    sUsed = solver_colisionGroup_map[sUsed];


//                     std::cerr  <<"#############################Dynamic: " << currentNode->getName() << " : " << sUsed->getName() <<  "\n";
                //Add the main solver, and link it to the vertex describing the dynamic object
                solverNode_dynamicObject_map.insert(std::make_pair(sUsed, currentVertex) );
                solverUsed.insert(sUsed);

            }
        }





        //We deal with all the solvers one by one
        std::set< Node* >::iterator it;
        for (it=solverUsed.begin(); it != solverUsed.end(); it++)
        {

            Node* currentSolver=*it;

            static_cast<BglNode*>(currentSolver)->clearInteractionForceFields();

            std::string animationName;
            std::set< InteractionForceField* > IFFAdded;

//                 std::cerr << "Solving with " << (*it)->getName() << "\n";
            typedef std::multimap< Node*,Hvertex>::iterator dynamicObjectIterator;
            std::pair<dynamicObjectIterator,dynamicObjectIterator> rangeSolver;
            rangeSolver = solverNode_dynamicObject_map.equal_range( currentSolver );


            if (rangeSolver.first == rangeSolver.second) continue; //No dynamic object, so no need to animate

            //Link the solver to the master Node
            Hvertex solverVertex =h_node_vertex_map[currentSolver];
            addHedge( masterVertex,solverVertex);
            //Add all the dynamic object depending of the current solver
            for (dynamicObjectIterator itDynObj=rangeSolver.first; itDynObj!=rangeSolver.second; itDynObj++)
            {
//                     std::cerr << "\tAdding " << h_vertex_node_map[itDynObj->second]->getName() << "\n";
                animationName += h_vertex_node_map[itDynObj->second]->getName() + " " ;
                BglSimulation::Hedge eH=addHedge( solverVertex,itDynObj->second ); // add nodes


                Rvertex objR= r_node_vertex_map[ h_vertex_node_map[ itDynObj->second ] ];
                for (unsigned int IFFIndex=0; IFFIndex<group.second.size(); ++IFFIndex)
                {
//                         std::cerr << "Interaction ; "
//                                   << h_vertex_node_map[group.second[IFFIndex].v1]->getName() << " and "
//                                   << h_vertex_node_map[group.second[IFFIndex].v2]->getName() << "\n";
                    Rvertex ir1 = r_node_vertex_map[ h_vertex_node_map[ group.second[IFFIndex].v1 ] ];
                    Rvertex ir2 = r_node_vertex_map[ h_vertex_node_map[ group.second[IFFIndex].v2 ] ];
                    if (IFFAdded.find(group.second[IFFIndex].iff) == IFFAdded.end() &&
                        (isIFFinNode(ir1,objR) || isIFFinNode(ir2,objR)) )
                    {
//                             std::cerr << " %%%%%%%%%%%%%%%%%%%%%%%%% ADDING INTERACTION\n";
                        currentSolver->moveObject(group.second[IFFIndex].iff);
                        IFFAdded.insert(group.second[IFFIndex].iff);
                    }
                }
            }


//                   std::cerr << "\n\nStatus before Animating:\n";
//                    print(masterNode);
#ifdef DUMP_VISITOR_INFO
            simulation::Visitor::printComment(std::string("Animate ") + animationName );
#endif
            // animate this interaction group
            masterNode->animate(dt);

            //Remove the Interaction ForceFields
            std::set<InteractionForceField*>::iterator IFF_iterator;
            for (IFF_iterator=IFFAdded.begin(); IFF_iterator!=IFFAdded.end(); IFF_iterator++) currentSolver->removeObject(*IFF_iterator);

            //Remove the link between all the vertices connected to the solver
            removeHedge( masterVertex, solverVertex);
            //Clear the solverVertex from all the dynamic objects
            clear_vertex(solverVertex, hgraph);
        }
        for (unsigned int i=0; i<staticObjectAdded.size(); ++i) remove_edge(staticObjectAdded[i],hgraph);


//             std::cerr << "\n\nFinal Status:\n\n\n";
//             clear_vertex(masterVertex, hgraph);
//             print(masterNode);
    }



    //TODO: add the interactions to the graph
    for ( HvertexVector::iterator i=hroots.begin(), iend=hroots.end(); i!=iend; i++ ) addHedge( masterVertex,*i);


    for ( unsigned i=0; i<interactionGroups.size(); i++ )
    {
        // add the vertices and the interactions
        InteractionGroup& group = interactionGroups[i];
        for (unsigned int IFFIndex=0; IFFIndex<group.second.size(); ++IFFIndex)
        {
            masterNode->moveObject(group.second[IFFIndex].iff);
        }
    }

}



/// TODO: adapt the AnimateVisitor to BGL
void BglSimulation::animate(Node* root, double dt)
{
    dt = root->getContext()->getDt();

    clear_vertex( masterVertex, hgraph );

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
        std::cerr << h_vertex_node_map[*iter.first]->getName() << " " << in_degree (*iter.first,hgraph) << "\n";
        if ( isUsableRoot(h_vertex_node_map[*iter.first]) && in_degree (*iter.first,hgraph)==0 )
        {
            hroots.push_back(*iter.first);
            //                   cerr<<"node "<<h_vertex_node_map[*iter.first]->getName()<<" is a root"<<endl;
        }
    }
}

// after the collision detection, modification to the graph can have occured. We update the graph
void BglSimulation::updateGraph()
{

    std::set<Node*>::iterator it;
    for (it=nodeToDelete.begin(); it!=nodeToDelete.end(); it++) deleteNodeNow(*it);
    for (unsigned int i=0; i<nodeToAdd.size(); i++)             insertNewNode(nodeToAdd[i]);
    for (unsigned int i=0; i<edgeToAdd.size(); i++)             addNodeNow(edgeToAdd[i].first,edgeToAdd[i].second);
    for (unsigned int i=0; i<interactionToAdd.size(); i++)      addInteractionNow(interactionToAdd[i]);

    if (nodeToDelete.size() || nodeToAdd.size())
    {
        computeHroots();
        computeInteractionGraphAndConnectedComponents();
    }

    nodeToDelete.clear();
    nodeToAdd.clear();
    edgeToAdd.clear();
    interactionToAdd.clear();
}


void BglSimulation::computeBBox(Node* root, SReal* minBBox, SReal* maxBBox)
{
    sofa::simulation::Simulation::computeBBox(root,minBBox,maxBBox);

    for ( VisualVector::iterator i=visualModels.begin(), iend=visualModels.end(); i!=iend; i++ )
    {
        (*i).first->addBBox(minBBox,maxBBox);
    }
}

/// Toggle the showBehaviorModel flag of all systems
void BglSimulation::setShowBehaviorModels( bool b )
{
    for ( Hvpair iter=boost::vertices(hgraph); iter.first!=iter.second; iter.first++)
        h_vertex_node_map[*iter.first]->setShowBehaviorModels(b);
}
/// Toggle the showVisualModel flag of all systems
void BglSimulation::setShowVisualModels( bool b )
{
    for ( Hvpair iter=boost::vertices(hgraph); iter.first!=iter.second; iter.first++)
        h_vertex_node_map[*iter.first]->setShowVisualModels(b);
}
/// Toggle the showNormals flag of all systems
void BglSimulation::setShowNormals( bool b )
{
    for ( Hvpair iter=boost::vertices(hgraph); iter.first!=iter.second; iter.first++)
        h_vertex_node_map[*iter.first]->setShowNormals(b);
}

/// Display flags: Collision Models
void BglSimulation::setShowCollisionModels(bool val)
{
    for ( Hvpair iter=boost::vertices(hgraph); iter.first!=iter.second; iter.first++)
        h_vertex_node_map[*iter.first]->setShowCollisionModels(val);
}

/// Display flags: Bounding Collision Models
void BglSimulation::setShowBoundingCollisionModels(bool val)
{
    for ( Hvpair iter=boost::vertices(hgraph); iter.first!=iter.second; iter.first++)
        h_vertex_node_map[*iter.first]->setShowBoundingCollisionModels(val);
}

/// Display flags: Mappings
void BglSimulation::setShowMappings(bool val)
{
    for ( Hvpair iter=boost::vertices(hgraph); iter.first!=iter.second; iter.first++)
        h_vertex_node_map[*iter.first]->setShowMappings(val);
}

/// Display flags: Mechanical Mappings
void BglSimulation::setShowMechanicalMappings(bool val)
{
    for ( Hvpair iter=boost::vertices(hgraph); iter.first!=iter.second; iter.first++)
        h_vertex_node_map[*iter.first]->setShowMechanicalMappings(val);
}

/// Display flags: ForceFields
void BglSimulation::setShowForceFields(bool val)
{
    for ( Hvpair iter=boost::vertices(hgraph); iter.first!=iter.second; iter.first++)
        h_vertex_node_map[*iter.first]->setShowForceFields(val);
}

/// Display flags: InteractionForceFields
void BglSimulation::setShowInteractionForceFields(bool val)
{
    for ( Hvpair iter=boost::vertices(hgraph); iter.first!=iter.second; iter.first++)
        h_vertex_node_map[*iter.first]->setShowInteractionForceFields(val);
}

/// Display flags: WireFrame
void BglSimulation::setShowWireFrame(bool val)
{
    for ( Hvpair iter=boost::vertices(hgraph); iter.first!=iter.second; iter.first++)
        h_vertex_node_map[*iter.first]->setShowWireFrame(val);
}

void BglSimulation::setVisualModel( VisualModel* v, Mapping* m)
{
    visualModels.push_back( std::make_pair(v,m) );
}

void BglSimulation::draw(Node* , helper::gl::VisualParameters*)
{
    if (!masterNode) return;
    // 	cerr<<"begin BglSimulation::glDraw()"<<endl;
    masterNode->glDraw();

    for (HvertexVector::iterator i=hroots.begin(), iend=hroots.end(); i!=iend; i++ )
    {
        h_vertex_node_map[*i]->glDraw();
    }

    for ( VisualVector::iterator i=visualModels.begin(), iend=visualModels.end(); i!=iend; i++ )
    {
        //cerr<<"draw visual model, mapped from "<<(*i).second->getFrom()->getName()<<" to "<<(*i).second->getTo()->getName()<<endl;
        (*i).second->updateMapping();
        (*i).first->drawVisual();
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

    for (unsigned int i=0; i<visualModels.size(); ++i)
    {
        delete visualModels[i].first;
        delete visualModels[i].second;
    }
    visualModels.clear();


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


