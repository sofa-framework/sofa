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

#ifndef SOFA_SIMULATION_BGL_BGLGRAPHMANAGER_H
#define SOFA_SIMULATION_BGL_BGLGRAPHMANAGER_H

#define BOOST_NO_HASH

#include <sofa/simulation/bgl/bgl.h>
#include <sofa/simulation/common/Node.h>
#include <boost/graph/adjacency_list.hpp>
#include <sofa/core/collision/Pipeline.h>

namespace sofa
{
namespace simulation
{
namespace bgl
{
class BglNode;

class SOFA_SIMULATION_BGL_API BglGraphManager
{
public:


    /** @name Hierarchical graph
    This graph (hgraph) reflects the sofa mechanical mapping hierarchy.
    The vertices contains sofa simulation nodes, while the edges contain nothing.
    The (mechanical) mappings are not in the edges because in future work we want to allow a single mapping to read input from several parents. Thus, the mappings remain associated with the Nodes as usual.

    A dual graph (rgraph) is used for convenience.
    It is the same as the hierarchical graph, but its edges are in the opposite directions.
    (BTW, do we really need it ?)
    */



    /// @{
    ///Definition of the Graph

    // Vertices
    //Defining a new property tag so that we can attach a Sofa Node to a vertex of the boost graph
    struct bglnode_t
    {
        typedef boost::vertex_property_tag kind;
    };
    //The Vertex Property: as we will need to process to removals, and additions at any time possible, we need to store the vertices inside a list.
    //As a list in Boost doesn't have vertex index, we add an index as property to the vertices.
    typedef
    boost::property< bglnode_t            , Node*, //Property linking a vertex of the boost graph to a Sofa Node
          boost::property< boost::vertex_index_t, unsigned int    //Property needed to be able to launch visitors. Each vertex must have a unique id
          > > VertexProperty;

    // Graph
    //As the graph is sparse, we use an adjacency_list to describe it.
    //The edges are stored in a vector (maybe use a list instead?)
    //bidirectional = oriented + each node stores its in-edges, allowing to easily detect roots (root = no in-edges)
    typedef boost::adjacency_list < boost::listS, boost::listS, boost::bidirectionalS, VertexProperty > Hgraph;
    typedef boost::graph_traits<Hgraph> HgraphTraits;
    /* 	typedef boost::reverse_graph<Hgraph> ReverseHgraph;  */

    //Defining some typedefs for convenience usage
    typedef HgraphTraits::vertex_descriptor Hvertex;
    typedef HgraphTraits::edge_descriptor   Hedge;

    typedef HgraphTraits::vertex_iterator   HvertexIterator;
    typedef HgraphTraits::edge_iterator     HedgeIterator;

    typedef std::pair<HvertexIterator,HvertexIterator> Hvpair;

    typedef helper::vector<Hvertex> HvertexVector;


    //Defining the Property Map: link between a vertex of the BglGraph and one of its property
    // hvertex->sofa node
    typedef boost::property_map<Hgraph, bglnode_t>::type  H_vertex_node_map;
    //Inverse map, that we must keep up to date
    // sofa node->hvertex
    typedef std::map<Node*, Hvertex> H_node_vertex_map;
    ///@}

    //-----------------------------------------------------------------------------------
    // reverse graph: hgraph with vertices in the opposite direction.
    typedef
    boost::property< bglnode_t            , Node*, //Property linking a vertex of the boost graph to a Sofa Node
          boost::property< boost::vertex_index_t, unsigned int    //Property needed to be able to launch visitors. Each vertex must have a unique id
          /*           boost::property< parentnodes_t        , bool ,  */
          /*           boost::property< childnodes_t         , bool     */
          /*           > > */ > > RVertexProperty;

    // Graph
    typedef boost::adjacency_list < boost::listS, boost::listS, boost::bidirectionalS, RVertexProperty > Rgraph;
    typedef boost::graph_traits<Rgraph> RgraphTraits;

    typedef Rgraph::vertex_descriptor Rvertex;
    typedef Rgraph::edge_descriptor   Redge;

    typedef RgraphTraits::vertex_iterator     RvertexIterator;
    typedef RgraphTraits::edge_iterator     RedgeIterator;

    typedef std::pair<RvertexIterator,RvertexIterator> Rvpair;

    typedef boost::property_map<Rgraph, bglnode_t>::type  R_vertex_node_map; // rvertex->sofa node
    typedef std::map<Node*, Rvertex> R_node_vertex_map;                    //  sofa node->rvertex

    /** @name interaction graph
    This auxiliary graph represents the interactions between the simulated objects.
    It is created temporarily and used to build the interaction groups.
    Contrary to the hierarchical graph, it is not oriented.
    */
    ///@{
    // same Vertices as Hgraph
    // Edges
    struct interaction_t
    {
        typedef boost::edge_property_tag kind;
    };
    typedef boost::property<interaction_t, core::objectmodel::BaseObject*> IEdgeProperty;
    // Graph
    typedef boost::adjacency_list < boost::vecS, boost::vecS, boost::undirectedS, VertexProperty, IEdgeProperty > Igraph;
    typedef Igraph::vertex_descriptor Ivertex;
    typedef Igraph::edge_descriptor Iedge;
    typedef std::pair<Igraph::vertex_iterator,Igraph::vertex_iterator> Ivpair;

    typedef boost::property_map<Igraph, bglnode_t>::type  I_vertex_node_map;           // ivertex->sofa node
    typedef std::map<Node*, Ivertex>     I_node_vertex_map;                              // sofa node -> ivertex
    typedef boost::property_map<Igraph, interaction_t>::type  I_edge_interaction_map;  // iedge->sofa interaction force field
    typedef std::map<core::objectmodel::BaseObject*, Iedge> I_interaction_edge_map;              // sofa interaction force field->iedge
    ///@}

    /** @name interactions
    Interactions can be defined at any level of the hgraph.
    Based on a list of interactions, we build interaction groups : lists of objects and the interactions between them.
    Each object is represented by its root in the hierarchical graph.
    */
    ///@{
    struct InteractionData
    {
        Node *n1;
        Node *n2;
        core::objectmodel::BaseObject* iff;
        InteractionData( Node *r1, Node *r2, core::objectmodel::BaseObject* i ) : n1(r1), n2(r2), iff(i) {}
    };

    struct Interaction
    {
        Hvertex v1;
        Hvertex v2;
        core::objectmodel::BaseObject* iff;
        Interaction( Hvertex r1, Hvertex r2, core::objectmodel::BaseObject* i ) : v1(r1), v2(r2), iff(i) {}
    };
    typedef helper::vector<Interaction> Interactions;
    typedef helper::vector<InteractionData> InteractionsData;
    typedef std::pair< HvertexVector, helper::vector<Interaction> > InteractionGroup; ///< maximum set of nodes which interact together, along with the interactions between them
    typedef helper::vector<InteractionGroup> InteractionGroups;
    ///@}

public:

    BglGraphManager();

    //*********************************************************************************
    //Singleton class
    static BglGraphManager* getInstance()
    {
        static BglGraphManager instance;
        return &instance;
    }

    //*********************************************************************************
    //Basic operations to modify the Graph
    void addVertex(BglNode* node);
    void removeVertex(BglNode* node);
    void clearVertex(BglNode *node);
    void addEdge(Node *p, Node*c);
    void removeEdge( Node* p, Node* c );
    void addInteraction( Node* n1, Node* n2, core::objectmodel::BaseObject* );
    void removeInteraction( core::objectmodel::BaseObject* );

    //*********************************************************************************
    //Basic operations to consult the Graph
    //Find all the vertices connected to the node as in edges of the graph (ex: find children of a node)
    template <typename Container>
    void getParentNodes(Container &data, const Node* node);
    //Find all the vertices connected to the node as out edges of the graph (ex: find parents of a node)
    template <typename Container>
    void getChildNodes(Container &data, const Node* node);
    //Fill the container with the roots of the graph
    template <typename Container>
    void getRoots(Container &data);

    //*********************************************************************************
    //Visitors implementation
    /// breadth visit from the given vertex
    void breadthFirstVisit( const Node* n, Visitor&, core::objectmodel::BaseContext::SearchDirection);
    /// depth search in the whole scene
    void depthFirstSearch(  Visitor&, core::objectmodel::BaseContext::SearchDirection);
    /// depth visit starting from the given vertex
    void depthFirstVisit( const Node* n,  Visitor&, core::objectmodel::BaseContext::SearchDirection);



    //*********************************************************************************
    void update();
    void reset();
    void printDebug();

protected:

    /** Compute the interaction graph and the connected components, based on interactions and hroots
    */
    void computeInteractionGraphAndConnectedComponents();

    /** Determine if we have to recompute the interaction graph
     */
    bool needToComputeInteractions();

    /** Compute the Roots of the graphs
     */
    void computeRoots();


    inline Rvertex convertHvertex2Rvertex(Hvertex v);
    inline Hvertex convertRvertex2Hvertex(Rvertex v);

    template <typename Graph, typename VertexMap>
    inline void  addVertex(BglNode *node, Graph &g, VertexMap &map);

    template <typename Graph, typename VertexMap>
    inline void  removeVertex(BglNode *node, Graph &g, VertexMap &vmap);

    template <typename Graph>
    inline typename Graph::edge_descriptor  addEdge( typename Graph::vertex_descriptor p, typename Graph::vertex_descriptor c,Graph &g);

    template <typename Graph>
    inline void clearVertex( typename Graph::vertex_descriptor v, Graph &g);

    template <typename Graph>
    inline Node *getNode( typename Graph::vertex_descriptor v, Graph &g);

    template <typename Container, typename Graph>
    inline void getInVertices(Container &data, typename Graph::vertex_descriptor v,  Graph &g);

    template <typename Container, typename Graph>
    inline void getOutVertices(Container &data, typename Graph::vertex_descriptor v,  Graph &g);





    HvertexVector hroots;           ///< the roots of the forest
    Hvertex visualRoot;
//        BglNode*   visualNode;

    Interactions interactions;            ///< interactions between nodes at at any hierarchical levels
    Interactions previousInteractions;    ///< interactions between nodes at at any hierarchical levels at the previous time step
    InteractionGroups interactionGroups;  ///< all the objects and interactions, in independent groups which can be processed separately

    Hgraph hgraph;             ///< the directed acyclic graph representing system dependencies (edges correspond to mappings)
    Rgraph rgraph;             ///< The reverse graph

    H_node_vertex_map h_node_vertex_map;     ///< sofa node->hvertex
    R_node_vertex_map r_node_vertex_map;     ///< sofa node->rvertex



    //DEBUG
    template <typename Graph>
    void printVertices(Graph &g);
    template <typename Graph>
    void printEdges(Graph &g);

};
}
}
}
#endif
