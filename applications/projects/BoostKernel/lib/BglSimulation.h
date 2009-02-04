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
    typedef sofa::core::componentmodel::behavior::BaseMechanicalMapping MechanicalMapping;
    typedef sofa::core::componentmodel::behavior::InteractionForceField InteractionForceField;
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
    HvertexVector hroots;      ///< the roots of the forest
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
    void  removeRedge(Hvertex parent, Hvertex child);
    /// @}


    /** @name visual models
        The sofa visual models and their mappings are separated from the mechanical mapping hierarchy.
        This makes the hierarchy graph more homogeneous and hopefully easier to process.
    */
    /// @{
    typedef vector<std::pair<VisualModel*,Mapping*> > VisualVector;

    VisualVector visualModels;
    /// @}


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
    typedef boost::property<interaction_t, InteractionForceField*> IEdgeProperty;
    // Graph
    typedef ::boost::adjacency_list < ::boost::vecS, ::boost::vecS, ::boost::undirectedS, VertexProperty, IEdgeProperty > Igraph;
    typedef Igraph::vertex_descriptor Ivertex;
    typedef Igraph::edge_descriptor Iedge;
    typedef std::pair<Igraph::vertex_iterator,Igraph::vertex_iterator> Ivpair;

    typedef ::boost::property_map<Igraph, bglnode_t>::type  I_vertex_node_map;           // ivertex->sofa node
    typedef std::map<Node*, Ivertex>     I_node_vertex_map;                              // sofa node -> ivertex
    typedef ::boost::property_map<Igraph, interaction_t>::type  I_edge_interaction_map;  // iedge->sofa interaction force field
    typedef std::map<InteractionForceField*, Iedge> I_interaction_edge_map;              // sofa interaction force field->iedge

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
    struct Interaction
    {
        Hvertex v1;
        Hvertex v2;
        InteractionForceField* iff;
        Interaction( Hvertex r1, Hvertex r2, InteractionForceField* i ) : v1(r1), v2(r2), iff(i) {}
    };
    typedef vector<Interaction> Interactions;
    typedef std::pair< HvertexVector, vector<Interaction> > InteractionGroup; ///< maximum set of nodes which interact together, along with the interactions between them
    typedef vector<InteractionGroup> InteractionGroups;

    Interactions interactions;            ///< interactions between nodes at at any hierarchical levels
    InteractionGroups interactionGroups;  ///< all the objects and interactions, in independent groups which can be processed separately

    /** Compute the interaction graph and the connected components, based on interactions and hroots
     */
    void computeInteractionGraphAndConnectedComponents();

    ///@}

    /// Make the correspondance between a node in the mechanical graph and the solver nodes.
    std::map< Node*, Node*> node_solver_map;


    /// @name High-level interface
    /// @{
    BglSimulation();

    /// Create a graph edge, parent to child, and attach the MechanicalMapping to the child
    void setMechanicalMapping(BglNode* parent, BglNode* child, MechanicalMapping* );

    /// Add a visual model to the scene, attached by a Mapping.
    /// They are not inserted in a scene graph, but in a separated container.
    /// The Mapping needs not be attached to a Node.
    void setVisualModel( VisualModel*, Mapping* );

    /// Add an interaction
    void addInteraction( Node* n1, Node* n2, InteractionForceField* );
    /// Remove an interaction
    //void removeInteraction( InteractionForceField* );

    /// Load a file
    Node* load(const char* filename);

    /// Load a file
    void unload(Node* root);

    /// Create a graph node and attach a new Node to it, then return the Node
    Node* newNode(const std::string& name="");

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


    /// Toggle the showBehaviorModel flag of all systems
    void setShowBehaviorModels( bool );

    /// Toggle the showVisualModel flag of all systems
    void setShowVisualModels( bool );

    /// Toggle the showNormals flag of all systems
    void setShowNormals( bool );

    /// Display flags: Collision Models
    void setShowCollisionModels(bool val);

    /// Display flags: Bounding Collision Models
    void setShowBoundingCollisionModels(bool val);

    /// Display flags: Mappings
    void setShowMappings(bool val);

    /// Display flags: Mechanical Mappings
    void setShowMechanicalMappings(bool val);

    /// Display flags: ForceFields
    void setShowForceFields(bool val);

    /// Display flags: InteractionForceFields
    void setShowInteractionForceFields(bool val);

    /// Display flags: WireFrame
    void setShowWireFrame(bool val);

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

    /*
       ///The collision models belong to the hgraph because they are mechanically bound to the objects.
       ///Additionally, they are referenced in an auxiliary data structure to ease the collision detection.

       Hgraph cgraph; ///< Hierarchical graph which contains all the nodes which have a collision model, organized in a flat hierarchy.
       Node* collisionNode;
       Hvertex collisionVertex; ///< Root of the collision graph. Contains the collision detection and response components
       H_vertex_node_map  c_vertex_node_map; ///< access the nodes of the collision graph
       CollisionPipeline* collisionPipeline;

    */
    /// @}


};

Simulation* getSimulation();
}
}
}

#endif


