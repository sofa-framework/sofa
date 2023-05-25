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
#include <sofa/core/objectmodel/Base.h>
#include <sofa/core/fwd.h>

namespace sofa::simulation
{
    class Node;
    typedef sofa::core::sptr<Node> NodeSPtr;
}

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

    SOFA_ATTRIBUTE_DISABLED("v21.06 (PR#1730)", "v21.12", "Use sofa::core::visual::DisplayFlags instead.")
    typedef DeprecatedAndRemoved DisplayFlags;

    Simulation();
    virtual ~Simulation();

    Simulation(const Simulation& n) = delete;
    Simulation& operator=(const Simulation& n) = delete;

    /// Print all object in the graph
    SOFA_ATTRIBUTE_DEPRECATED_SIMULATION_PRINT()
    virtual void print(Node* root);

    /// Initialize the objects
    virtual void init(Node* root);

    ///Init a node without changing the context of the simulation.
    virtual void initNode(Node* node);


    /// Execute one timestep. If dt is 0, the dt parameter in the graph will be used
    virtual void animate(Node* root, SReal dt=0.0);

    /// Update the Visual Models: triggers the Mappings
    virtual void updateVisual(Node* root);

    /// Reset to initial state
    virtual void reset(Node* root);

    /// Initialize the textures
    virtual void initTextures(Node* root);

    /// Update contexts. Required before drawing the scene if root flags are modified.
    virtual void updateContext(Node* root);

    /// Update contexts. Required before drawing the scene if root flags are modified.
    virtual void updateVisualContext(Node* root);

    /** Compute the bounding box of the scene.
     * If init is set to "true", then minBBox and maxBBox will be initialised to a default value
     * @warning MechanicalObjects with showObject member set to false are ignored
     * @sa computeTotalBBox(Node* root, SReal* minBBox, SReal* maxBBox)
     */
    virtual void computeBBox(Node* root, SReal* minBBox, SReal* maxBBox, bool init=true);

    /** Compute the bounding box of the scene.
     * Includes all objects, may they be displayed or not.
     * @sa computeBBox(Node* root, SReal* minBBox, SReal* maxBBox, bool init=true)
     * @deprecated
     */
    virtual void computeTotalBBox(Node* root, SReal* minBBox, SReal* maxBBox);

    /// Render the scene
    virtual void draw(sofa::core::visual::VisualParams* vparams, Node* root);

    /// Export a scene to an OBJ 3D Scene
    virtual void exportOBJ(Node* root, const char* filename, bool exportMTL = true);

    /// Print all object in the graph in XML format
    SOFA_ATTRIBUTE_DEPRECATED_SIMULATION_EXPORTXML()
    virtual void exportXML(Node* root, const char* fileName=nullptr);

    /// Print all objects in the graph in the given file (format is given by the filename extension)
    virtual void exportGraph(Node* root, const char* filename=nullptr);

    /// Dump the current state in the given stream
    virtual void dumpState( Node* root, std::ofstream& out );

    /// Load a scene from a file
    virtual NodeSPtr load(const std::string& /* filename */, bool reload = false, const std::vector<std::string>& sceneArgs = std::vector<std::string>(0));

    /// Unload a scene from a Node.
    virtual void unload(NodeSPtr root);

    /// create a new graph(or tree) and return its root node.
    virtual NodeSPtr createNewGraph(const std::string& name)=0;//Todo replace newNode method

    /// creates and returns a new node.
    virtual NodeSPtr createNewNode(const std::string& name)=0;

    /// Can the simulation handle a directed acyclic graph?
    virtual bool isDirectedAcyclicGraph() = 0;

private:

    // use sofa::simulation::setSimulation and sofa::simulation::getSimulation instead
    static DeprecatedAndRemoved theSimulation;
};
} // namespace sofa::simulation

MSG_REGISTER_CLASS(sofa::simulation::Simulation, "Simulation")
