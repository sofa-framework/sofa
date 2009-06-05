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
#ifndef SOFA_SIMULATION_BGL_BGLSIMULATION_H
#define SOFA_SIMULATION_BGL_BGLSIMULATION_H

#include "BglGraphManager.h"

#include <sofa/simulation/common/Simulation.h>
#include <sofa/core/componentmodel/behavior/BaseMechanicalMapping.h>
#include <sofa/core/componentmodel/behavior/InteractionForceField.h>
#include <sofa/core/componentmodel/behavior/InteractionConstraint.h>


#include <sofa/core/componentmodel/behavior/MasterSolver.h>
#include <sofa/core/componentmodel/behavior/OdeSolver.h>
#include <sofa/core/componentmodel/behavior/LinearSolver.h>

#include <sofa/core/BaseMapping.h>
#include <sofa/core/VisualModel.h>
#include <sofa/helper/gl/VisualParameters.h>
#include <sofa/helper/vector.h>
#include <map>

using sofa::core::componentmodel::behavior::MasterSolver;
using sofa::core::componentmodel::behavior::OdeSolver;
using sofa::core::componentmodel::behavior::LinearSolver;

namespace sofa
{
namespace simulation
{
namespace bgl
{

//      class BglGraphManager;

using sofa::helper::vector;
using sofa::simulation::Node;

/// SOFA scene implemented using bgl graphs and with high-level modeling and animation methods.
class BglSimulation : public sofa::simulation::Simulation
{
public:
    typedef BglGraphManager::Hgraph Hgraph;
    typedef BglGraphManager::Hvertex Hvertex;
    typedef BglGraphManager::Rvertex Rvertex;
    typedef BglGraphManager::Hedge Hedge;
    typedef BglGraphManager::Redge Redge;
    typedef BglGraphManager::InteractionData InteractionData;

    /* 	typedef BglNode Node;     ///< sofa simulation node */
    typedef sofa::core::componentmodel::behavior::InteractionForceField InteractionForceField;
    typedef sofa::core::componentmodel::behavior::InteractionConstraint InteractionConstraint;
    typedef sofa::core::componentmodel::collision::Pipeline CollisionPipeline;
    typedef sofa::core::BaseMapping Mapping;
    typedef sofa::core::VisualModel VisualModel;

    ///@}

    /// If a Collision Group is created with a new solver responsible for the animation, we need to update the "node_solver_map"
    void setSolverOfCollisionGroup(Node* solverNode, Node* solverOfCollisionGroup);

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

    /* 	/// Add an interaction */
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

    void reset ( Node* root );

    // Node dynamical access
    /// Add a node to as the child of another
    void addNode(BglNode* parent, BglNode* child);

    /// Delete a graph node and all the edges, and entries in map
    void deleteNode( Node* n);


    /// Initialize all the nodes and edges depth-first
    void init();


    /// Animate all the nodes depth-first
    void animate(Node* root, double dt=0.0);

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
    /* 	BglNode* masterNode; */
    /* 	Hvertex masterVertex; */

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


    BglGraphManager graphManager;
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
