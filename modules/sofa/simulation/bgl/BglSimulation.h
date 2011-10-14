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

#include <sofa/simulation/bgl/BglGraphManager.h>

#include <sofa/simulation/common/Simulation.h>

namespace sofa
{
namespace simulation
{
namespace bgl
{
/// SOFA scene implemented using bgl graphs and with high-level modeling and animation methods.
class SOFA_SIMULATION_BGL_API BglSimulation : public sofa::simulation::Simulation
{
public:
    SOFA_CLASS(BglSimulation, sofa::simulation::Simulation);

    /// @name High-level interface
    /// @{
    BglSimulation();

    /// Load a file, and update the graph
    Node::SPtr load(const char* filename);
    void unload(Node::SPtr root);

    ///create a new graph(or tree) and return its root node
    Node::SPtr createNewGraph(const std::string& name="");




    void reset ( Node* root );

    /// Initialize all the nodes and edges depth-first
    void init(Node* root);
    /// @}
};

SOFA_SIMULATION_BGL_API Simulation* getSimulation();
}
}
}

#endif
