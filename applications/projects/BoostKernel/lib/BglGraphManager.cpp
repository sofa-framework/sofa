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
// C++ Implementation: BglGraphManager
//
// Description:
//
//
// Author: Francois Faure in The SOFA team </www.sofa-framework.org>, (C) 2008
//
// Copyright: See COPYING file that comes with this distribution
//
//

#include "BglGraphManager.h"
#include "BglNode.h"

#include "dfs_adapter.h"
#include "dfv_adapter.h"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/vector_property_map.hpp>


#include <sofa/helper/vector.h>

#include <algorithm>
#include <map>

#include <sofa/simulation/common/CollisionVisitor.h>
#include <sofa/simulation/common/PrintVisitor.h>

namespace sofa
{
namespace simulation
{
namespace bgl
{

BglGraphManager::BglGraphManager()
{
    h_vertex_node_map = get( bglnode_t(), hgraph);
    r_vertex_node_map = get( bglnode_t(), rgraph);
    // 	c_vertex_node_map = get( bglnode_t(), cgraph);

    // The animation control overloads the solvers of the scene
    Node *n=newNodeNow("masterNode");
    masterNode= dynamic_cast<BglNode*>(n);
    masterVertex = h_node_vertex_map[masterNode];

    collisionPipeline=NULL;
};



void BglGraphManager::insertHierarchicalGraph()
{
    /// Initialize all the nodes using a depth-first visit
    for (HvertexVector::iterator i=hroots.begin(), iend=hroots.end(); i!=iend; i++ )
    {
        //Link all the nodes to the MasterNode, in ordre to propagate the Context
        Hgraph::edge_descriptor e1; bool found=false;
        tie(e1,found) = edge(masterVertex, *i,hgraph);
        if (!found) addEdge( masterVertex, *i); // add nodes
    }
}


BglGraphManager::Rvertex BglGraphManager::convertHvertex2Rvertex(Hvertex v)
{
    return r_node_vertex_map[ h_vertex_node_map[ v ] ];
}
BglGraphManager::Hvertex BglGraphManager::convertRvertex2Hvertex(Rvertex v)
{
    return h_node_vertex_map[ r_vertex_node_map[ v ] ];
}



/// depth search in the whole scene
void BglGraphManager::dfs( Visitor& visit )
{
    dfs_adapter vis(&visit,h_vertex_node_map);
    boost::depth_first_search( hgraph, boost::visitor(vis) );
}
/// depth search starting from the given vertex, and prunable
void BglGraphManager::dfv( Hvertex v, Visitor& vis )
{
    boost::vector_property_map<boost::default_color_type> colors( boost::num_vertices(hgraph) );
    dfv_adapter dfsv(&vis,this,h_vertex_node_map );
    boost::depth_first_visit(
        hgraph,
        boost::vertex(v, hgraph),
        dfsv,
        colors,
        dfsv
    );
}


Node* BglGraphManager::getNodeFromHvertex(Hvertex h)
{
    Hvpair vIterator=vertices(hgraph);
    if (std::find(vIterator.first, vIterator.second, h) != vIterator.second) return h_vertex_node_map[h];
    else return NULL;
}
Node* BglGraphManager::getNodeFromRvertex(Rvertex r)
{
    Rvpair vIterator=vertices(rgraph);
    if (std::find(vIterator.first, vIterator.second, r) != vIterator.second) return r_vertex_node_map[r];
    else return NULL;
}

void BglGraphManager::addEdge( Hvertex p, Hvertex c )
{
    if (p == c) return;
    addHedge(p,c);
    addRedge(convertHvertex2Rvertex(c),convertHvertex2Rvertex(p));
}

BglGraphManager::Hedge BglGraphManager::addHedge( Hvertex p, Hvertex c )
{
    std::pair<Hedge, bool> e =  add_edge(  p,c,hgraph);
    assert(e.second);
    return e.first;
}

BglGraphManager::Redge BglGraphManager::addRedge( Rvertex p, Rvertex c )
{
    std::pair<Redge, bool> e =  add_edge(  p,c,rgraph);
    assert(e.second);
    return e.first;
}


void BglGraphManager::removeEdge( Hvertex p, Hvertex c )
{
    if (p == c) return;
    removeHedge(p,c);
    removeRedge(convertHvertex2Rvertex(c),convertHvertex2Rvertex(p));
}

void BglGraphManager::removeHedge( Hvertex p, Hvertex c )
{
    remove_edge(  p,c,hgraph);
}

void BglGraphManager::removeRedge( Rvertex p, Rvertex c )
{
    remove_edge(  p,c,rgraph);
}


void BglGraphManager::addEdge( Node* from, Node* to )
{
    edgeToAdd.insert(std::make_pair(from, to));
}

void BglGraphManager::removeEdge( Node* from, Node* to )
{
    removeHedge( h_node_vertex_map[from], h_node_vertex_map[to] );
    removeRedge( r_node_vertex_map[to], r_node_vertex_map[from] );
}

void BglGraphManager::deleteVertex( Hvertex v)
{
    deleteHvertex(v);
    deleteRvertex(convertHvertex2Rvertex(v));
}

void BglGraphManager::deleteHvertex( Hvertex vH)
{
//         std::cerr << "Delete Now Node " << n << " : \n";
//         std::cerr << n->getName() << " : " << h_node_vertex_map[n] << "\n";
    Node *node = h_vertex_node_map[vH];
    clear_vertex(vH, hgraph);
    h_node_vertex_map.erase(node);
    h_vertex_node_map[vH]=NULL;
    //Cannot remove vertex: it changes all the vertices of the graph !!!
//         remove_vertex(vH, hgraph);
}


void BglGraphManager::deleteRvertex( Rvertex vR)
{
//         std::cerr << "Delete Now Node " << n << " : \n";
//         std::cerr << n->getName() << " : " << h_node_vertex_map[n] << "\n";
    Node *node = r_vertex_node_map[vR];
    clear_vertex(vR, rgraph);
    r_node_vertex_map.erase(node);
    r_vertex_node_map[vR]=NULL;

    //Cannot remove vertex: it changes all the vertices of the graph !!!
//         remove_vertex(vR, rgraph);
}


void BglGraphManager::clearVertex( Hvertex v )
{
    clear_vertex(v,hgraph);
    clear_vertex(convertHvertex2Rvertex(v),rgraph);
}

/// Update the graph with all the operation stored in memory: add/delete node, add interactions...
void BglGraphManager::updateGraph()
{
//         std::cerr << "node:" <<nodeToAdd.size() << " : edge:" << edgeToAdd.size() << " : interaction:" << interactionToAdd.size() << "\n";
    for (std::set<Hvertex>::iterator it=vertexToDelete.begin(); it!=vertexToDelete.end(); it++) deleteVertex(*it);
    for (unsigned int i=0; i<nodeToAdd.size() ; i++)               insertNewNode(nodeToAdd[i]);
    std::set< std::pair< Node*, Node*> >::iterator itEdge;
    for (itEdge=edgeToAdd.begin(); itEdge!=edgeToAdd.end(); itEdge++)
    {
        if (itEdge->first != masterNode && !h_node_vertex_map[itEdge->first])
        {
            addNode(static_cast<BglNode*>(itEdge->first));
        }
        if (itEdge->first != masterNode && !h_node_vertex_map[itEdge->second])
        {
            addNode(static_cast<BglNode*>(itEdge->second));
        }
        addEdge(h_node_vertex_map[ itEdge->first],h_node_vertex_map[ itEdge->second]);
    }
    for (unsigned int i=0; i<interactionToAdd.size(); i++)         addInteractionNow(interactionToAdd[i]);

    if (vertexToDelete.size() || nodeToAdd.size() || edgeToAdd.size()) computeRoots();


    vertexToDelete.clear();
    nodeToAdd.clear();
    edgeToAdd.clear();
    interactionToAdd.clear();
}

void BglGraphManager::update()
{
    updateGraph();

    if (needToComputeInteractions()) computeInteractionGraphAndConnectedComponents();

}



void BglGraphManager::addNode(BglNode *node)
{
    // Each BglNode needs a vertex in hgraph
    Hvertex hnode =  add_vertex( hgraph);
    node->graphManager = this;
    node->vertexId = hnode;
    node->graph = &hgraph;

    h_vertex_node_map[hnode] = node;
    h_node_vertex_map[node] = hnode;

    Rvertex rnode = add_vertex( rgraph );
    r_vertex_node_map[rnode] = node;
    r_node_vertex_map[node] = rnode;
}

Node* BglGraphManager::newNodeNow(const std::string& name)
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

void BglGraphManager::reset()
{
    for ( unsigned i=0; i<interactionGroups.size(); i++ )
    {
        interactionGroups[i].second.clear();
    }
    solver_colisionGroup_map.clear();

    vertexToDelete.clear();
    nodeToAdd.clear();
    edgeToAdd.clear();
    interactionToAdd.clear();
}

void BglGraphManager::clear()
{
    updateGraph();

    {
        Hgraph::vertex_iterator h_vertex_iter, h_vertex_iter_end;
        for (tie(h_vertex_iter,h_vertex_iter_end) = vertices(hgraph); h_vertex_iter != h_vertex_iter_end; ++h_vertex_iter)
        {
            remove_vertex(*h_vertex_iter, hgraph);
        }
    }
    {
        Rgraph::vertex_iterator r_vertex_iter, r_vertex_iter_end;
        for (tie(r_vertex_iter,r_vertex_iter_end) = vertices(rgraph); r_vertex_iter != r_vertex_iter_end; ++r_vertex_iter)
        {
            remove_vertex(*r_vertex_iter, rgraph);
        }
    }

    solver_colisionGroup_map.clear();
    nodeSolvers.clear();
    nodeGroupSolvers.clear();

    hgraph.clear();
    h_node_vertex_map.clear();
    rgraph.clear();
    r_node_vertex_map.clear();
    collisionPipeline=NULL;

    h_vertex_node_map = get( bglnode_t(), hgraph);
    r_vertex_node_map = get( bglnode_t(), rgraph);

    // The animation control overloads the solvers of the scene
    Node *n=newNodeNow("masterNode");
    masterNode= dynamic_cast<BglNode*>(n);
    masterVertex = h_node_vertex_map[masterNode];
}


/// Create a graph node and attach a new Node to it, then return the Node
Node* BglGraphManager::newNode(const std::string& name)
{
    BglNode* s  = new BglNode(this,name);
    nodeToAdd.push_back(s);
    return s;
}

void BglGraphManager::insertNewNode(Node *n)
{
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


/// Add an OdeSolver working inside a given Node
void BglGraphManager::addSolver(BaseObject* s,Node* n)
{
    nodeSolvers.insert(n);
}

void BglGraphManager::setSolverOfCollisionGroup(Node* solverNode, Node* solverOfCollisionGroup)
{
    solver_colisionGroup_map[solverNode] = solverOfCollisionGroup;
    nodeGroupSolvers.insert(solverOfCollisionGroup);
//         std::cerr << "Inserting " << solverOfCollisionGroup->getName() << "\n";
}


/// Add a node to as the child of another
void BglGraphManager::addNode(BglNode* parent, BglNode* child)
{
//           std::cerr << "ADD Node " << parent->getName() << "-->" << child->getName() << " : \n";
//           if (parent != masterNode)

//           std::cerr << "Add Node : push an edge " << parent->getName() << "-->" << child->getName() << " : \n";
    edgeToAdd.insert(std::make_pair(parent,child));
}
void BglGraphManager::deleteNode( Node* n)
{
//         std::cerr << "Delete Node " << n << " : \n";
    vertexToDelete.insert(h_node_vertex_map[n]);
}




void BglGraphManager::addInteraction( Node* n1, Node* n2, BaseObject* iff )
{
    interactionToAdd.push_back( InteractionData(n1, n2, iff) );
}

void BglGraphManager::addInteractionNow( InteractionData &i )
{
    if (i.n1 != i.n2) interactions.push_back( Interaction(h_node_vertex_map[i.n1], h_node_vertex_map[i.n2],i.iff));
}

void BglGraphManager::removeInteraction( BaseObject* iff )
{
    Interactions::iterator it;
    for (it=interactions.begin(); it!=interactions.end(); ++it)
    {
        if (it->iff == iff)
        {
//                 ((Node*)(iff->getContext()))->removeObject(iff);
            interactions.erase(it);
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
    typedef vector<BglGraphManager::Rvertex> Rleaves;
    Rleaves& leaves; // use external data, since internal data seems corrupted after the visit (???)
    find_leaves( Rleaves& l ):leaves(l) {}
    void discover_vertex( BglGraphManager::Rvertex v, const BglGraphManager::Rgraph& g )
    {
        if ( out_degree (v,g)==0 )  // leaf vertex
        {
            leaves.push_back(v);
        }
    }

};
}



bool BglGraphManager::needToComputeInteractions()
{
    bool need=false;
    if (interactions.size() != previousInteractions.size())
    {
        need=true;
    }
    else
    {
        for ( unsigned int i=0; i<interactions.size(); ++i)
        {
            if (interactions[i].iff != previousInteractions[i].iff)
            {
                need = true;
                break;
            }
        }
    }
    previousInteractions = interactions;
    return need;
}

/**
Data: hroots, interactions
 Result: interactionGroups
    */
void BglGraphManager::computeInteractionGraphAndConnectedComponents()
{

    ///< the interaction graph
    Igraph igraph;
    I_vertex_node_map      i_vertex_node_map;
    I_node_vertex_map      i_node_vertex_map;
    ///< iedge->sofa interaction force field
    I_edge_interaction_map i_edge_interaction_map;
    ///< sofa interaction force field->iedge
    I_interaction_edge_map i_interaction_edge_map;


    i_vertex_node_map = get( bglnode_t(), igraph);
    i_edge_interaction_map = get( interaction_t(), igraph );

    // create the interaction graph vertices (root nodes only)
    igraph.clear();
//          std::cerr << "hroots:";
    for ( HvertexVector::iterator i=hroots.begin(), iend=hroots.end(); i!=iend; i++ )
    {
        Ivertex iv = add_vertex( igraph );
        Node* n = h_vertex_node_map[*i];
//              std::cerr << n->getName() << ", ";
        i_vertex_node_map[iv] = n;
        i_node_vertex_map[n] = iv;
    }
//          cerr << endl;
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
//             cerr<<"find all the roots associated with the interaction from "<<h_vertex_node_map[(*i).v1]->getName()<<" to "<<h_vertex_node_map[(*i).v2]->getName()<<endl;

        // find all the roots associated with the given interaction
        vector<BglGraphManager::Rvertex> leaves;
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
//         	     cerr<<"end connected components"<<endl;
//        for( unsigned i=0; i<interactionGroups.size(); i++ )
//           {
//             cerr<<"interaction group (roots only): "<<endl;
//             cerr<<"- nodes = ";
//             for( unsigned j=0; j<interactionGroups[i].first.size(); j++ )
//               {
//                 Node* root = h_vertex_node_map[ interactionGroups[i].first[j] ];
//                 cerr<< root->getName() <<", ";
//               }
//             cerr<<endl;
//           }
}



void BglGraphManager::computeRoots()
{

    /// find the roots in hgraph
    hroots.clear();
//             visualroots.clear();
//             collisionroots.clear();
    for ( Hvpair iter=boost::vertices(hgraph); iter.first!=iter.second; iter.first++)
    {

        Node *vNode=h_vertex_node_map[*iter.first];
        if (!vNode) continue;



        unsigned int degree = in_degree (*iter.first,hgraph);
        if (degree==0 && *iter.first != masterVertex)
        {
//                     std::cerr << degree << " : " << degree << " ## " << vNode->getName() << "\n";
            hroots.push_back(*iter.first);
        }

//                 if (isVisualRoot(vNode)) visualroots.push_back(*iter.first);
//                 if (isCollisionRoot(vNode)) collisionroots.push_back(*iter.first);
    }

}

/// Perform the collision detection
void BglGraphManager::collisionStep(Node* root, double dt)
{
    if (collisionPipeline)
    {
        masterNode->addObject( collisionPipeline );
        CollisionVisitor act;
        masterNode->doExecuteVisitor(&act);
        masterNode->removeObject( collisionPipeline );
    }
}
/** Data: interaction groups
    Result: nodes updated
*/
void BglGraphManager::mechanicalStep(Node* root, double dt)
{
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("MechanicalStep");
#endif
    clearVertex(masterVertex);
    update();

    for ( unsigned i=0; i<interactionGroups.size(); i++ )
    {
        // remove previous children and interactions

        clearVertex( masterVertex );

        // add the vertices and the interactions
        InteractionGroup& group = interactionGroups[i];
        //Find the different objects
        std::set< Hvertex > staticObjectAdded; //objects without solver
        std::set< Hvertex > animatedObjectAdded; //objects animated by a solver
        std::set< Hvertex > solverUsed;


        std::string staticObjectName;
        for ( unsigned j=0; j<group.first.size(); j++ )
        {
            Hvertex currentVertex = group.first[j];
            Node   *currentNode   = h_vertex_node_map[ currentVertex ];
            if (!currentNode) continue;
            //No solver Found: we link it to the masterNode
            if (nodeSolvers.find( currentNode )      == nodeSolvers.end() &&
                nodeGroupSolvers.find( currentNode ) == nodeGroupSolvers.end())
            {
                addHedge( masterVertex, currentVertex); // add static object
                staticObjectAdded.insert(currentVertex);
                staticObjectName += currentNode->getName() + " ";
            }
            else
            {
                Node *sUsed=currentNode;
                //Verify if the current solver is not controled by a collision group
                if (solver_colisionGroup_map.find(sUsed) != solver_colisionGroup_map.end())
                {
                    sUsed = solver_colisionGroup_map[sUsed];
                    addHedge(h_node_vertex_map[sUsed], currentVertex);
                    //Add the main solver, and link it to the vertex describing the dynamic object
                }

                if (nodeSolvers.find( currentNode ) != nodeSolvers.end())
                    animatedObjectAdded.insert(currentVertex);

                solverUsed.insert(h_node_vertex_map[sUsed]);
            }
        }
        //No object to animate
        if (staticObjectAdded.empty() && animatedObjectAdded.empty() ) continue;

        //We deal with all the solvers one by one
        std::set< Hvertex >::iterator it;

        for (it=solverUsed.begin(); it != solverUsed.end(); it++)
        {
            std::string animationName;
            Hvertex solverVertex =*it;
            Node* currentSolver=h_vertex_node_map[solverVertex];
            // animate this interaction group
            addHedge( masterVertex,solverVertex);
            {
#ifdef SOFA_DUMP_VISITOR_INFO
                simulation::Visitor::printComment(std::string("Collision Group Animate ") + staticObjectName + currentSolver->getName() );
#endif
                masterNode->animate(dt);
                removeHedge( masterVertex, solverVertex);
            }
        }

        if (animatedObjectAdded.empty() // || !hasCollisionGroupManager
           )
        {
#ifdef SOFA_DUMP_VISITOR_INFO
            simulation::Visitor::printComment(std::string("Animate ") + staticObjectName );
#endif
            masterNode->animate(dt);
        }
    }


#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("MechanicalStep");
#endif
}


}
}
}
