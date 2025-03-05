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
#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/graph/config.h>
#include <sofa/simulation/fwd.h>
#include <memory>



namespace sofa::simulation::graph
{

/** Main controller of the scene.
Defines how the scene is inited at the beginning, and updated at each time step.
Derives from BaseObject in order to model the parameters as Datas, which makes their edition easy in the GUI.
 */
class SOFA_SIMULATION_GRAPH_API DAGSimulation: public Simulation
{
public:
    DAGSimulation();
    ~DAGSimulation() override; // this is a terminal class

    /// create a new graph(or tree) and return its root node.
    virtual NodeSPtr createNewGraph(const std::string& name) override;

    /// creates and returns a new node.
    virtual NodeSPtr createNewNode(const std::string& name) override;

    /// Can the simulation handle a directed acyclic graph?
    bool isDirectedAcyclicGraph() override { return true; }
};

/** Get the (unique) simulation which controls the scene.
Automatically creates one if no Simulation has been set.
 */
SOFA_ATTRIBUTE_DISABLED_DAGSIMULATION_GETSIMULATION()
Simulation* getSimulation() = delete;
} // namespace sofa::simulation::graph
