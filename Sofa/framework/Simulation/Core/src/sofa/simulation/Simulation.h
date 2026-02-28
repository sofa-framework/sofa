/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once

#include <sofa/simulation/config.h>
#include <sofa/simulation/fwd.h>
#include <sofa/helper/logging/Messaging.h>

namespace sofa::simulation
{

/** Main controller of the scene.
    Defines how the scene is inited at the beginning, and updated at each time step.
    Derives from Base in order to use smart pointers and model the parameters as Datas, which makes their edition easy in the GUI.
 */
class SOFA_SIMULATION_CORE_API Simulation
{
public:

    using SPtr = std::shared_ptr<Simulation>;

    Simulation();
    virtual ~Simulation();

    Simulation(const Simulation& n) = delete;
    Simulation& operator=(const Simulation& n) = delete;

    /// create a new graph(or tree) and return its root node.
    virtual NodeSPtr createNewGraph(const std::string& name);//Todo replace newNode method

    /// creates and returns a new node.
    virtual NodeSPtr createNewNode(const std::string& name);

    /// Can the simulation handle a directed acyclic graph?
    virtual bool isDirectedAcyclicGraph();
};
} // namespace sofa::simulation

MSG_REGISTER_CLASS(sofa::simulation::Simulation, "Simulation")
