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

#include "BglGraphManager.inl"
#include "BglNode.h"

#include "bfs_adapter.h"
#include "dfs_adapter.h"
#include "dfv_adapter.h"

#include <boost/graph/connected_components.hpp>
#include <boost/vector_property_map.hpp>

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
};


//----------------------------------------------------------------------------------
// Methods to modify the Graph

//**************************************************
// addVertex keeping both graph up-to-date        //
//**************************************************
void BglGraphManager::addVertex(BglNode *node)
{
    //Add a vertex to the bgl graph
    addVertex(node,hgraph, h_node_vertex_map);
    addVertex(node,rgraph, r_node_vertex_map);
}

//Insert a new vertex inside a graph boost
template < typename Graph, typename VertexMap>
void  BglGraphManager::addVertex(BglNode *node, Graph &g, VertexMap &vmap)
{
    typedef typename Graph::vertex_descriptor Vertex;
    Vertex v=add_vertex(g);
    boost::put(bglnode_t(),             g, v, node);
    boost::put(boost::vertex_index_t(), g, v, node->getId());
    vmap[node]=v;
}




//**************************************************
// deleteVertex keeping both graph up-to-date     //
//**************************************************
//Insert a new vertex inside a graph boost


void BglGraphManager::removeVertex(BglNode *node)
{
    //Add a vertex to the bgl graph
    removeVertex(node,hgraph, h_node_vertex_map);
    removeVertex(node,rgraph, r_node_vertex_map);
}


template < typename Graph, typename VertexMap>
void  BglGraphManager::removeVertex(BglNode *node, Graph &g, VertexMap &vmap)
{
    typedef typename Graph::vertex_descriptor Vertex;
    Vertex v=vmap[node];
    clear_vertex(v, g);
    vmap.erase(node);
    remove_vertex(v, g);
}


//**************************************************
// addEdge keeping both graph up-to-date          //
//**************************************************
void BglGraphManager::addEdge( Node* from, Node* to )
{
    //Verify if the node we need to link is already present in the graph
    if (h_node_vertex_map.find(from) == h_node_vertex_map.end()) addVertex(static_cast<BglNode*>(from));
    if (h_node_vertex_map.find(to)   == h_node_vertex_map.end()) addVertex(static_cast<BglNode*>(to));

    Hvertex hfrom=h_node_vertex_map[from];
    Hvertex hto  =h_node_vertex_map[to];
    addEdge( hfrom, hto, hgraph);

    Rvertex rfrom=r_node_vertex_map[from];
    Rvertex rto  =r_node_vertex_map[to];
    addEdge( rto, rfrom, rgraph);
}


template <typename Graph>
typename Graph::edge_descriptor BglGraphManager::addEdge( typename Graph::vertex_descriptor p, typename Graph::vertex_descriptor c,Graph &g)
{
    std::pair<typename Graph::edge_descriptor, bool> e =  add_edge(p,c,g);
    assert(e.second);
    return e.first;
}


//**************************************************
// removeEdge keeping both graph up-to-date       //
//**************************************************
void BglGraphManager::removeEdge( Node* from, Node* to )
{
    if (h_node_vertex_map.find(from) == h_node_vertex_map.end() ||
        h_node_vertex_map.find(to) == h_node_vertex_map.end() )
    {
//             std::cerr << "Error !!\n";
        return;
    }
//         std::cerr << "Remove Edge from " << from->getName() << " to " << to->getName() << "\n";
    Hvertex hfrom=h_node_vertex_map[from];
    Hvertex hto  =h_node_vertex_map[to];
    remove_edge(hfrom, hto, hgraph);

    Rvertex rfrom=r_node_vertex_map[from];
    Rvertex rto  =r_node_vertex_map[to];
    remove_edge(rto, rfrom, rgraph);
}

//**************************************************
// clearVertex                                    //
//**************************************************

void BglGraphManager::clearVertex( BglNode* node)
{
    clearVertex(h_node_vertex_map[node],hgraph);
    clearVertex(r_node_vertex_map[node],rgraph);
}

template <typename Graph>
void BglGraphManager::clearVertex( typename Graph::vertex_descriptor v, Graph &g)
{
    clear_vertex(v, g);
}


void BglGraphManager::addInteraction( Node* n1, Node* n2, BaseObject* iff )
{
    if (n1 == n2) return;
    interactions.push_back( Interaction(h_node_vertex_map[n1], h_node_vertex_map[n2],iff));
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
//         std::cerr << iff << "@" << iff->getName() << " : ########################################## NO REMOVAL\n";
}

//----------------------------------------------------------------------------------
// Methods to consult the Graph

//TODO: understand why we need this "magic number" to use the visitors with the Boost:
#define HACK_MAGIC_NUMBER 200


//**************************************************
// Breadth First Visit                            //
//**************************************************

template< typename Visitor, typename Graph,  typename SystemMap>
struct launchBreadthFirstVisit
{

    typedef typename boost::graph_traits<Graph>::vertex_descriptor Vertex;
    typedef typename boost::property_map<Graph, BglGraphManager::bglnode_t>::type  Vertex_Node_Map;

    launchBreadthFirstVisit( Visitor &v, Graph &g, SystemMap &s):  visitor(v), graph(g),systemMap(s)
    {
    }

    void operator()(Node *node)
    {
        BglGraphManager::ColorMap colors;
        colors.resize(HACK_MAGIC_NUMBER*boost::num_vertices(graph));
        boost::queue<Vertex> queue;

        Vertex_Node_Map nodeMap=get(BglGraphManager::bglnode_t(),graph);
        bfs_adapter<Graph> bfsv(&visitor, nodeMap);
        Vertex v=systemMap[node];
        boost::breadth_first_visit(graph,
                v,
                queue,
                bfsv,
                make_iterator_property_map(colors.begin(),boost::get(boost::vertex_index, graph) )
                                  );
    }

    Visitor &visitor;
    Graph &graph;
    SystemMap &systemMap;
};


/// breadth first visit starting from the given vertex, and prunable
void BglGraphManager::breadthFirstVisit( const Node *constNode, Visitor& visit, core::objectmodel::BaseContext::SearchDirection dir )
{
    using core::objectmodel::BaseContext;
    Node* n = const_cast<Node*>(constNode);

    if (h_node_vertex_map.find(n) == h_node_vertex_map.end())
        addVertex((BglNode*)(n));

    switch(dir)
    {
    case BaseContext::SearchDown:
    {
        launchBreadthFirstVisit<Visitor, Hgraph, H_node_vertex_map> launcher(visit, hgraph, h_node_vertex_map);
        break;
    }
    case BaseContext::SearchUp:
    {
        launchBreadthFirstVisit<Visitor, Rgraph, R_node_vertex_map> launcher(visit, rgraph, r_node_vertex_map);
        break;
    }
    case BaseContext::SearchRoot:
    {
        std::cerr << "breadth First Visit not implemented yet for Root direction" << std::endl;
        break;
    }
    case BaseContext::Local:
    {
        std::cerr << "depthFirstVisit cannot be used with a Local Direction" << std::endl;
    }
    }
}


//**************************************************
// Depth First Search                             //
//**************************************************

/// depth search in the whole scene
void BglGraphManager::depthFirstSearch( Visitor& visit, core::objectmodel::BaseContext::SearchDirection dir)
{
    using core::objectmodel::BaseContext;
    switch(dir)
    {
    case BaseContext::SearchRoot:
    case BaseContext::SearchDown:
    {
        H_vertex_node_map systemMap=get(bglnode_t(),hgraph);
        dfs_adapter<Hgraph> vis(&visit,systemMap);
        boost::depth_first_search( hgraph, boost::visitor(vis) );
        break;
    }
    case BaseContext::SearchUp:
    {
        R_vertex_node_map systemMap=get(bglnode_t(),rgraph);
        dfs_adapter<Rgraph> vis(&visit,systemMap);
        boost::depth_first_search( rgraph, boost::visitor(vis) );
        break;
    }
    case BaseContext::Local:
    {
        std::cerr << "depthFirstSearch cannot be used with a Local Direction" << std::endl;
        break;
    }
    }
}


template< typename Visitor, typename Graph,  typename SystemMap>
struct launchDepthFirstVisit
{

    typedef typename boost::graph_traits<Graph>::vertex_descriptor Vertex;
    typedef typename boost::property_map<Graph, BglGraphManager::bglnode_t>::type  Vertex_Node_Map;

    launchDepthFirstVisit( Visitor &v, Graph &g, SystemMap &s):  visitor(v), graph(g),systemMap(s)
    {
    }

    void operator()(Node *node)
    {
        BglGraphManager::ColorMap colors;
        colors.resize(HACK_MAGIC_NUMBER*boost::num_vertices(graph));

        Vertex_Node_Map nodeMap=get(BglGraphManager::bglnode_t(),graph);
        dfv_adapter<Graph> dfsv(&visitor, nodeMap);
        Vertex v=systemMap[node];
        boost::depth_first_visit(graph,
                v,
                dfsv,
                make_iterator_property_map(colors.begin(),boost::get(boost::vertex_index, graph) ),
                dfsv
                                );
    }

    Visitor &visitor;
    Graph &graph;
    SystemMap &systemMap;
};


//**************************************************
// Depth First Visit                              //
//**************************************************

/// depth search starting from the given vertex, and prunable
void BglGraphManager::depthFirstVisit( const Node *constNode, Visitor& visit, core::objectmodel::BaseContext::SearchDirection dir )
{
    using core::objectmodel::BaseContext;
    Node* n = const_cast<Node*>(constNode);
    if (h_node_vertex_map.find(n) == h_node_vertex_map.end())
        addVertex((BglNode*)(n));

//         std::cerr << visit.getClassName() << "  " << visit.getInfos() << " depthFirstVisitor with " << constNode->getName() << "\n";

    switch(dir)
    {
    case BaseContext::SearchDown:
    {
        launchDepthFirstVisit<Visitor, Hgraph, H_node_vertex_map>(visit, hgraph, h_node_vertex_map)(n);
        break;
    }
    case BaseContext::SearchUp:
    {
        launchDepthFirstVisit<Visitor, Rgraph, R_node_vertex_map>(visit, rgraph, r_node_vertex_map)(n);
        break;
    }
    case BaseContext::SearchRoot:
    {

        std::vector< Node* > roots;
        getRoots(roots);
        launchDepthFirstVisit<Visitor, Hgraph, H_node_vertex_map> launcher(visit, hgraph, h_node_vertex_map);
        std::for_each(roots.begin(), roots.end(),launcher);
        break;
    }
    case BaseContext::Local:
    {
        std::cerr << "depthFirstVisit cannot be used with a Local Direction" << std::endl;
    }
    }
}

//----------------------------------------------------------------------------------



void BglGraphManager::update()
{
    computeRoots();
    if (needToComputeInteractions()) computeInteractionGraphAndConnectedComponents();
}



void BglGraphManager::reset()
{
    for ( unsigned i=0; i<interactionGroups.size(); i++ )
    {
        interactionGroups[i].second.clear();
    }
}


void BglGraphManager::clear()
{
    hgraph.clear();
    printVertices(hgraph);
    h_node_vertex_map.clear();
    rgraph.clear();
    r_node_vertex_map.clear();
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
namespace
{
template< typename Graph>
class find_leaves: public ::boost::bfs_visitor<>
{
    typedef typename Graph::vertex_descriptor Vertex;
public:
    typedef vector<Vertex> VertexLeaves;
    VertexLeaves& leaves; // use external data, since internal data seems corrupted after the visit (???)
    find_leaves(VertexLeaves& l ):leaves(l) {}
    void discover_vertex( Vertex v, const Graph& g )
    {
        if ( out_degree (v,g)==0 )  // leaf vertex
        {
            leaves.push_back(v);
        }
    }

};

}
void BglGraphManager::computeInteractionGraphAndConnectedComponents()
{

    return ;
#if 0
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
        Node* n = getNode(*i,hgraph);
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
    std::cerr << "Interactions : " << interactions.size() << "!!!!!!!!!!\n";
    for ( Interactions::iterator i=interactions.begin(), iend=interactions.end(); i!=iend; i++ )
    {
        std::cerr << getNode((*i).v1, hgraph)->getName() << " and " << getNode((*i).v2, hgraph)->getName()  << " : " << (*i).iff->getName() << std::endl;
    }
    for ( Interactions::iterator i=interactions.begin(), iend=interactions.end(); i!=iend; i++ )
    {
//             cerr<<"find all the roots associated with the interaction from "<<h_vertex_node_map[(*i).v1]->getName()<<" to "<<h_vertex_node_map[(*i).v2]->getName()<<endl;
        // find all the roots associated with the given interaction
        vector<Rvertex> leaves;
        find_leaves<Rgraph> visit(leaves);
        ColorMap colors;
        colors.resize(boost::num_vertices(rgraph));
        boost::queue<Rvertex> queue;
        Rvertex v1 = convertHvertex2Rvertex( (*i).v1 );
        Rvertex v2 = convertHvertex2Rvertex( (*i).v2 );

        boost::breadth_first_visit(rgraph,
                v1,
                queue,
                visit,
                make_iterator_property_map(colors.begin(),boost::get(boost::vertex_index, rgraph) )
                                  );

        boost::breadth_first_visit(rgraph,
                v2,
                queue,
                visit,
                make_iterator_property_map(colors.begin(),boost::get(boost::vertex_index, rgraph) )
                                  );

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
        typedef find_leaves<Rgraph>::VertexLeaves::iterator LeavesIterator;
        for ( LeavesIterator l=visit.leaves.begin(), lend=visit.leaves.end(); l!=lend; l++ )
        {
            for ( LeavesIterator m=l++; m!=lend; m++ )
            {
                if ( *l != *m )
                {
                    std::pair<Iedge,bool> e = add_edge( i_node_vertex_map[getNode(*l,rgraph)],i_node_vertex_map[getNode(*m,rgraph)], igraph );
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
#endif
}



void BglGraphManager::computeRoots()
{
    /// find the roots in hgraph
    hroots.clear();
    HvertexIterator it_begin, it_end;
    for (boost::tie(it_begin, it_end)=vertices(hgraph); it_begin!=it_end; ++it_begin)
    {
        unsigned int degree = in_degree (*it_begin,hgraph);

        if (degree==0)
        {
            hroots.push_back(*it_begin);
        }
    }
}

BglGraphManager::Rvertex BglGraphManager::convertHvertex2Rvertex(Hvertex v)
{
    return r_node_vertex_map[ getNode(v,hgraph) ];
}
BglGraphManager::Hvertex BglGraphManager::convertRvertex2Hvertex(Rvertex v)
{
    return h_node_vertex_map[ getNode(v,rgraph) ];
}



void BglGraphManager::printDebug()
{
    std::cerr << "******************************************************" << std::endl;
    std::cerr << "State of Hgraph" << std::endl;
    printVertices(hgraph);
    printEdges(hgraph);
    std::cerr << "State of Rgraph" << std::endl;
    printVertices(rgraph);
    printEdges(rgraph);
    std::cerr << "******************************************************" << std::endl;
    std::cerr << std::endl << std::endl;
}

//*****************************************************
// DEBUG
//*****************************************************

template <typename Graph>
void BglGraphManager::printVertices(Graph &g)
{
    typedef typename Graph::vertex_iterator VertexIterator;
    VertexIterator it, it_end;
    for (boost::tie(it, it_end)=vertices(g); it!=it_end; ++it)
    {
        std::cout << getNode(*it, g)->getName() << "@" << *it << "  ";
    }
    std::cout << std::endl;
}


template <typename Graph>
void BglGraphManager::printEdges(Graph &g)
{

    typedef typename Graph::edge_iterator EdgeIterator;
    EdgeIterator it, it_end;
    for (boost::tie(it, it_end)=edges(g); it!=it_end; ++it)
    {
        std::cout << getNode(source(*it,g), g)->getName()  << "->"
                << getNode(target(*it,g), g)->getName()  << "  ";
    }
    std::cout << std::endl;
}
}
}
}
