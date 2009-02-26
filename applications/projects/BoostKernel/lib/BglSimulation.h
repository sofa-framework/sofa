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
// C++ Interface: BglSimulation
//
// Description:
//
//
// Author: Francois Faure in The SOFA team </www.sofa-framework.org>, (C) 2008
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef BglSimulation_h
#define BglSimulation_h

#include <sofa/simulation/common/Simulation.h>
#include <sofa/core/componentmodel/behavior/BaseMechanicalMapping.h>
#include <sofa/core/componentmodel/behavior/InteractionForceField.h>
#include <sofa/core/componentmodel/behavior/InteractionConstraint.h>
#include <sofa/core/componentmodel/collision/Pipeline.h>


#include <sofa/core/componentmodel/behavior/MasterSolver.h>
#include <sofa/core/componentmodel/behavior/OdeSolver.h>
#include <sofa/core/componentmodel/behavior/LinearSolver.h>

#include <sofa/core/BaseMapping.h>
#include <sofa/core/VisualModel.h>
#include <sofa/helper/gl/VisualParameters.h>
#include <boost/graph/adjacency_list.hpp>
#include <sofa/helper/vector.h>
#include <map>

using sofa::core::componentmodel::behavior::MasterSolver;
using sofa::core::componentmodel::behavior::OdeSolver;
using sofa::core::componentmodel::behavior::LinearSolver;

namespace sofa
{
namespace simulation
{
namespace tree
{
class Visitor;
}
namespace bgl
{

class BglNode;

using sofa::helper::vector;
using sofa::simulation::Node;

/// SOFA scene implemented using bgl graphs and with high-level modeling and animation methods.
class BglSimulation : public sofa::simulation::Simulation
{
public:
    /* 	typedef BglNode Node;     ///< sofa simulation node */
    typedef sofa::core::componentmodel::behavior::InteractionForceField InteractionForceField;
    typedef sofa::core::componentmodel::behavior::InteractionConstraint InteractionConstraint;
    typedef sofa::core::componentmodel::collision::Pipeline CollisionPipeline;
    typedef sofa::core::BaseMapping Mapping;
    typedef sofa::core::VisualModel VisualModel;


    /** @name Hierarchical graph
        This graph (hgraph) reflects the sofa mechanical mapping hierarchy.
        The vertices contains sofa simulation nodes, while the edges contain nothing.
        The (mechanical) mappings are not in the edges because in future work we want to allow a single mapping to read input from several parents. Thus, the mappings remain associated with the Nodes as usual.

        A dual graph (rraph) is used for convenience.
        It is the same as the hierarchical graph, but its edges are in the opposite directions.
        (BTW, do we really need it ?)
    */
    /// @{
    // Vertices
    struct bglnode_t
    {
        typedef boost::vertex_property_tag kind;
    };
    typedef boost::property<bglnode_t, Node*> VertexProperty;

    // Graph
    typedef ::boost::adjacency_list < ::boost::vecS, ::boost::vecS, ::boost::bidirectionalS, VertexProperty > Hgraph; // bidirectional = oriented + each node stores its in-edges, allowing to easily detect roots (root = no in-edges)
    typedef Hgraph::vertex_descriptor Hvertex;
    typedef Hgraph::edge_descriptor Hedge;
    typedef std::pair<Hgraph::vertex_iterator,Hgraph::vertex_iterator> Hvpair;

    typedef vector<Hvertex> HvertexVector;
    typedef ::boost::property_map<Hgraph, bglnode_t>::type  H_vertex_node_map; // hvertex->sofa node
    /*    typedef ::boost::property_map<const Hgraph, bglnode_t>::type  H_vertex_node_const_map; // hvertex->sofa node*/
    typedef std::map<Node*, Hvertex> H_node_vertex_map;                    //  sofa node->hvertex

    Hgraph hgraph;             ///< the directed acyclic graph representing system dependencies (edges correspond to mappings)
    HvertexVector hroots;           ///< the roots of the forest
    HvertexVector visualroots;      ///< the roots of the visual graph
    HvertexVector collisionroots;   ///< the roots of the collision graph
    H_vertex_node_map h_vertex_node_map;           ///< hvertex->sofa node
    //H_vertex_node_const_map h_vertex_node_const_map;           ///< hvertex->sofa node
    H_node_vertex_map h_node_vertex_map;     ///< sofa node->hvertex

    //Hvertex addHvertex( sofa::simulation::Node* );
    Hedge addHedge(Hvertex parent, Hvertex child);
    void  removeHedge(Hvertex parent, Hvertex child);
    /// depth search in the whole scene
    void dfs( Visitor& );
    /// depth visit starting from the given vertex
    void dfv( Hvertex, Visitor& );

    // reverse graph: hgraph with vertices in the opposite direction.
    typedef boost::property<bglnode_t, Node*> RVertexProperty;
    // Graph
    typedef ::boost::adjacency_list < ::boost::vecS, ::boost::vecS, ::boost::bidirectionalS, RVertexProperty > Rgraph;
    typedef Rgraph::vertex_descriptor Rvertex;
    typedef Rgraph::edge_descriptor Redge;
    typedef std::pair<Rgraph::vertex_iterator,Rgraph::vertex_iterator> Rvpair;
    typedef boost::property_map<Rgraph, bglnode_t>::type  R_vertex_node_map; // rvertex->sofa node
    typedef std::map<Node*, Rvertex> R_node_vertex_map;                    //  sofa node->rvertex

    Rgraph rgraph;             ///< The reverse graph
    R_node_vertex_map r_node_vertex_map;     ///< sofa node->rvertex
    R_vertex_node_map r_vertex_node_map;  ///< rvertex->sofa node

    Redge addRedge(Rvertex parent, Rvertex child);
    void  removeRedge(Rvertex parent, Rvertex child);
    void  deleteVertex(Hvertex v);
    void  clearVertex(Hvertex v);
    /// @}

    Rvertex convertHvertex2Rvertex(Hvertex v);
    Hvertex convertRvertex2Hvertex(Rvertex v);
    void    addEdge(Hvertex p, Hvertex c);
    void    removeEdge(Hvertex p, Hvertex c);
    void    removeVertex(Hvertex p);


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
        typedef ::boost::edge_property_tag kind;
    };
    typedef boost::property<interaction_t, BaseObject*> IEdgeProperty;
    // Graph
    typedef ::boost::adjacency_list < ::boost::vecS, ::boost::vecS, ::boost::undirectedS, VertexProperty, IEdgeProperty > Igraph;
    typedef Igraph::vertex_descriptor Ivertex;
    typedef Igraph::edge_descriptor Iedge;
    typedef std::pair<Igraph::vertex_iterator,Igraph::vertex_iterator> Ivpair;

    typedef ::boost::property_map<Igraph, bglnode_t>::type  I_vertex_node_map;           // ivertex->sofa node
    typedef std::map<Node*, Ivertex>     I_node_vertex_map;                              // sofa node -> ivertex
    typedef ::boost::property_map<Igraph, interaction_t>::type  I_edge_interaction_map;  // iedge->sofa interaction force field
    typedef std::map<BaseObject*, Iedge> I_interaction_edge_map;              // sofa interaction force field->iedge

    Igraph igraph;                           ///< the interaction graph
    I_vertex_node_map      i_vertex_node_map;
    I_node_vertex_map      i_node_vertex_map;
    I_edge_interaction_map i_edge_interaction_map;           ///< iedge->sofa interaction force field
    I_interaction_edge_map i_interaction_edge_map;                       ///< sofa interaction force field->iedge

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
        BaseObject* iff;
        InteractionData( Node *r1, Node *r2, BaseObject* i ) : n1(r1), n2(r2), iff(i) {}
    };

    struct Interaction
    {
        Hvertex v1;
        Hvertex v2;
        BaseObject* iff;
        Interaction( Hvertex r1, Hvertex r2, BaseObject* i ) : v1(r1), v2(r2), iff(i) {}
    };
    typedef vector<Interaction> Interactions;
    typedef vector<InteractionData> InteractionsData;
    typedef std::pair< HvertexVector, vector<Interaction> > InteractionGroup; ///< maximum set of nodes which interact together, along with the interactions between them
    typedef vector<InteractionGroup> InteractionGroups;

    Interactions interactions;            ///< interactions between nodes at at any hierarchical levels
    Interactions previousInteractions;    ///< interactions between nodes at at any hierarchical levels at the previous time step
    InteractionGroups interactionGroups;  ///< all the objects and interactions, in independent groups which can be processed separately
    /** Compute the interaction graph and the connected components, based on interactions and hroots
     */
    void computeInteractionGraphAndConnectedComponents();

    /** Determine if we have to recompute the interaction graph
     */
    bool needToComputeInteractions();

    /** Compute the Roots of the graphs
     */
    void computeRoots();

    void insertHierarchicalGraph();
    void insertVisualGraph();
    void insertCollisionGraph();

    ///@}

    /// If a Collision Group is created with a new solver responsible for the animation, we need to update the "node_solver_map"
    void setSolverOfCollisionGroup(Node* solverNode, Node* solverOfCollisionGroup);

    /// Make the correspondance between a node in the mechanical graph and the solver nodes.
    std::map< Node*, Node*> solver_colisionGroup_map;
    /// Make the correspondance between a node in the mechanical graph and the solver nodes.
    std::set< Node*> nodeSolvers;
    std::set< Node*> nodeGroupSolvers;

    /// @name High-level interface
    /// @{
    BglSimulation();

    /// Method called when a MechanicalMapping is created.
    void setMechanicalMapping(Node *child, core::componentmodel::behavior::BaseMechanicalMapping *m);
    /// Method called when a MechanicalMapping is destroyed.
    void resetMechanicalMapping(Node *child, core::componentmodel::behavior::BaseMechanicalMapping *m);

    /// Method called when a MechanicalMapping is created.
    void setContactResponse(Node * parent, core::objectmodel::BaseObject* response);
    /// Method called when a MechanicalMapping is destroyed.
    void resetContactResponse(Node * parent, core::objectmodel::BaseObject* response);

    void clear();



    /// Add an interaction
    void addInteraction( Node* n1, Node* n2, BaseObject* );

    /// Add an interaction
    void addInteractionNow( InteractionData &i);

    /// Remove an interaction
    void removeInteraction( BaseObject* );

    /// Load a file
    Node* load(const char* filename);

    /// Load a file
    void unload(Node* root);

    /// Delayed Creation of a graph node and attach a new Node to it, then return the Node
    Node* newNode(const std::string& name="");

    /// Insert a node previously created, into the graph
    void insertNewNode(Node *n);

    /// Create a graph node and attach a new Node to it, then return the Node
    Node* newNodeNow(const std::string& name="");

    void initNewNode();

    void reset ( Node* root );

    // Node dynamical access
    /// Add a node to as the child of another
    void addNode(BglNode* parent, BglNode* child);

    /// Delete a graph node and all the edges, and entries in map
    void deleteNode( Node* n);

    /// Delete the hgraph node and all the edges, and entries in map
    void deleteHvertex( Hvertex n);
    /// Delete the rgraph node and all the edges, and entries in map
    void deleteRvertex( Rvertex n);


    /// Update the graph with all the operation stored in memory: add/delete node, add interactions...
    void updateGraph();

    bool isVisualRoot(Node* n);

    bool isCollisionRoot(Node* n);

    /// Initialize all the nodes and edges depth-first
    void init();


    /// Animate all the nodes depth-first
    void animate(Node* root, double dt=0.0);

    /// Animate all the nodes depth-first
    void mechanicalStep(Node* root, double dt=0.0);

    /// Compute the bounding box of the scene.
    void computeBBox(Node* root, SReal* minBBox, SReal* maxBBox);

    /// Render the scene
    void draw(Node* root, helper::gl::VisualParameters* params = NULL);


    /// Add a Solver working inside a given Node
    void addSolver(BaseObject*,Node* n);
    /// @}

    /** @name control
        The control node contains the Solver(s) and CollisionPipeline applied to the scene.
        Each independent set of objects is processed independently by this node.
        In future work, we may allow local overloading in nodes which contain a solver.


    */
    /// @{
    BglNode* masterNode;
    Hvertex masterVertex;
    CollisionPipeline* collisionPipeline;
    bool hasCollisionGroupManager;


    std::set   < Hvertex >                     vertexToDelete;
    std::set   < Node*   >                     externalDelete;
    std::vector< Node*   >                     nodeToAdd;
    std::set< std::pair<Node*,Node*> >         edgeToAdd;
    std::vector< InteractionData >             interactionToAdd;
    /// @}
    /// Methods to handle collision group:
    /// We create default solvers, that will eventually be used when two groups containing a solver will have to be managed at the same time
    Node* getSolverEulerEuler();
    Node* getSolverRungeKutta4RungeKutta4();
    Node* getSolverCGImplicitCGImplicit();
    Node* getSolverEulerImplicitEulerImplicit();
    Node* getSolverStaticSolver();
    Node* getSolverRungeKutta4Euler();
    Node* getSolverCGImplicitEuler();
    Node* getSolverCGImplicitRungeKutta4();
    Node* getSolverEulerImplicitEuler();
    Node* getSolverEulerImplicitRungeKutta4();
    Node* getSolverEulerImplicitCGImplicit();


protected:
    Node* solverEulerEuler;
    Node* solverRungeKutta4RungeKutta4;
    Node* solverCGImplicitCGImplicit;
    Node* solverEulerImplicitEulerImplicit;
    Node* solverStaticSolver;
    Node* solverRungeKutta4Euler;
    Node* solverCGImplicitEuler;
    Node* solverCGImplicitRungeKutta4;
    Node* solverEulerImplicitEuler;
    Node* solverEulerImplicitRungeKutta4;
    Node* solverEulerImplicitCGImplicit;



};

Simulation* getSimulation();
}
}
}

#endif


