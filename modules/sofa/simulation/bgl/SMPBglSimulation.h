/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_SIMULATION_BGL_SMPBGLSIMULATION_H
#define SOFA_SIMULATION_BGL_SMPBGLSIMULATION_H

#include <sofa/simulation/bgl/BglGraphManager.h>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/simulation/common/ChangeListener.h>
#include "Multigraph.h"

namespace sofa
{
namespace simulation
{
namespace bgl
{
class MainLoopTask;
/// SOFA scene implemented using bgl graphs and with high-level modeling and animation methods.
class SOFA_SIMULATION_BGL_API SMPBglSimulation : public sofa::simulation::Simulation
{
private:
    Iterative::Multigraph<MainLoopTask> *multiGraph;
    Iterative::Multigraph<MainLoopTask> *multiGraph2;
    common::ChangeListener *changeListener;
public:
    SOFA_CLASS(SMPBglSimulation, sofa::simulation::Simulation);

    /// @name High-level interface
    /// @{
    SMPBglSimulation();

    virtual ~SMPBglSimulation();

    /// Load a file, and update the graph
    Node* load(const char* filename);
    void unload(Node* root);

    ///create a new graph(or tree) and return its root node
    Node* createNewGraph(const std::string& name="");

    /// Delayed Creation of a graph node and attach a new Node to it, then return the Node
    Node* newNode(const std::string& name="");

    Node *getVisualRoot();

    void reset ( Node* root );

    /// Initialize all the nodes and edges depth-first
    void init(Node* root);
    /// @}

    /// Execute one timestep. If dt is 0, the dt parameter in the graph will be used
    virtual void animate(Node* root, double dt=0.0);
    virtual void generateTasks(Node* root, double dt=0.0);

protected:
    Node *visualNode;
    Data<bool> parallelCompile;
};

SOFA_SIMULATION_BGL_API Simulation* getSimulation();
}
}
}

#endif

