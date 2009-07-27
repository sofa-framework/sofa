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

namespace sofa
{
namespace simulation
{
namespace bgl
{

using sofa::helper::vector;
using sofa::simulation::Node;

/// SOFA scene implemented using bgl graphs and with high-level modeling and animation methods.
class BglSimulation : public sofa::simulation::Simulation
{
public:	typedef sofa::core::componentmodel::collision::Pipeline CollisionPipeline;

    ///@}

    /// If a Collision Group is created with a new solver responsible for the animation, we need to update the "node_solver_map"
    void setSolverOfCollisionGroup(Node* solverNode, Node* solverOfCollisionGroup);

    /// @name High-level interface
    /// @{
    BglSimulation();

    /// Load a file
    Node* load(const char* filename);

    /// Load a file
    void unload(Node* root);

    /// Delayed Creation of a graph node and attach a new Node to it, then return the Node
    Node* newNode(const std::string& name="");

    void clear();

    void reset ( Node* root );

    /// Initialize all the nodes and edges depth-first
    void init(Node* root);

    /// Animate all the nodes depth-first
    void animate(Node* root, double dt=0.0);
    /// @}

    BglGraphManager graphManager;


    /// /!\ Temporary implementation
    /// Need to be changed!
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
