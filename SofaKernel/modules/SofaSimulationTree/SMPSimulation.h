/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_SIMULATION_TREE_SMPSIMULATION_H
#define SOFA_SIMULATION_TREE_SMPSIMULATION_H

#include <sofa/simulation/Node.h>
#include <SofaSimulationTree/GNode.h>
#include <sofa/simulation/ChangeListener.h>
#include <SofaSimulationTree/TreeSimulation.h>
#include "Multigraph.h"

namespace sofa
{

namespace simulation
{

namespace tree
{

class MainLoopTask;

/** Main controller of the scene.
Defines how the scene is inited at the beginning, and updated at each time step.
Derives from BaseObject in order to model the parameters as Datas, which makes their edition easy in the GUI.
*/
class SOFA_SIMULATION_TREE_API SMPSimulation: public Simulation
{
private:
    Iterative::Multigraph<MainLoopTask> *multiGraph;
    Iterative::Multigraph<MainLoopTask> *multiGraph2;
    common::ChangeListener *changeListener;
public:
    SOFA_CLASS(SMPSimulation, Simulation);

    /** Load a scene from a file.
    Static method because at this point, the Simulation component is not yet created.
    If a Simulation component is found in the graph, then it is used.
    Otherwise, a default Simulation will be created at the first call to method getSimulation()
    This file can be a xml file or a script file which will generate a xml tree.
    */

    SMPSimulation();
    Node *getVisualRoot();
    virtual ~SMPSimulation();
    /// Initialize the objects
    virtual void init(Node* root);

    /// Execute one timestep. If dt is 0, the dt parameter in the graph will be used
    virtual void animate(Node* root, double dt=0.0);
    virtual void generateTasks(Node* root, double dt=0.0);

    /// create a new graph(or tree) and return its root node.
    virtual Node::SPtr createNewGraph(const std::string& name);

    /// creates and returns a new node.
    virtual Node::SPtr createNewNode(const std::string& name);

    /// Can the simulation handle a directed acyclic graph?
    virtual bool isDirectedAcyclicGraph() { return false; }

protected:
    Node *visualNode;
    Data<bool> parallelCompile; ///< Compile task graph in parallel
};


} // namespace tree

} // namespace simulation

} // namespace sofa

#endif
