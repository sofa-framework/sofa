/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_SIMULATION_TREE_TREESIMULATION_H
#define SOFA_SIMULATION_TREE_TREESIMULATION_H

#include <sofa/simulation/common/Simulation.h>
#include <sofa/SofaSimulation.h>
#include <memory>



namespace sofa
{

namespace simulation
{

namespace tree
{

/** Main controller of the scene.
Defines how the scene is inited at the beginning, and updated at each time step.
Derives from BaseObject in order to model the parameters as Datas, which makes their edition easy in the GUI.
 */
class SOFA_SIMULATION_TREE_API TreeSimulation: public Simulation
{
public:
    SOFA_CLASS(TreeSimulation, Simulation);

    TreeSimulation();
    ~TreeSimulation(); // this is a terminal class

    /// create a new graph(or tree) and return its root node.
    virtual Node::SPtr createNewGraph(const std::string& name);

    /// creates and returns a new node.
    virtual Node::SPtr createNewNode(const std::string& name);

    /// Can the simulation handle a directed acyclic graph?
    virtual bool isDirectedAcyclicGraph() { return false; }

//protected:
//	std::auto_ptr<Node> visualNode;
};

/** Get the (unique) simulation which controls the scene.
Automatically creates one if no Simulation has been set.
 */
SOFA_SIMULATION_TREE_API Simulation* getSimulation();
} // namespace tree

} // namespace simulation

} // namespace sofa

#endif
