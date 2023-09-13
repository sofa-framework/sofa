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

namespace node
{

/// Initialize the objects
void SOFA_SIMULATION_CORE_API initRoot(Node* root);
///Init a node without changing the context of the simulation.
void SOFA_SIMULATION_CORE_API init(Node* node);
/// Print all object in the graph in XML format
void SOFA_SIMULATION_CORE_API exportInXML(Node* root, const char* fileName);
/// Print all object in the graph
void SOFA_SIMULATION_CORE_API print(Node* root);
/// Update contexts. Required before drawing the scene if root flags are modified.
void SOFA_SIMULATION_CORE_API updateVisualContext(Node* root);
/// Execute one timestep. If dt is 0, the dt parameter in the graph will be used
void SOFA_SIMULATION_CORE_API animate(Node* root, SReal dt=0.0);
/// Update the Visual Models: triggers the Mappings
void SOFA_SIMULATION_CORE_API updateVisual(Node* root);
/// Reset to initial state
void SOFA_SIMULATION_CORE_API reset(Node* root);
/// Initialize the textures
void SOFA_SIMULATION_CORE_API initTextures(Node* root);
/// Update contexts. Required before drawing the scene if root flags are modified.
void SOFA_SIMULATION_CORE_API updateContext(Node* root);

/** Compute the bounding box of the scene.
 * If init is set to "true", then minBBox and maxBBox will be initialised to a default value
 * @warning MechanicalObjects with showObject member set to false are ignored
 * @sa computeTotalBBox(Node* root, SReal* minBBox, SReal* maxBBox)
 */
void SOFA_SIMULATION_CORE_API computeBBox(Node* root, SReal* minBBox, SReal* maxBBox, bool init=true);

/** Compute the bounding box of the scene.
 * Includes all objects, may they be displayed or not.
 * @sa computeBBox(Node* root, SReal* minBBox, SReal* maxBBox, bool init=true)
 * @deprecated
 */
void SOFA_SIMULATION_CORE_API computeTotalBBox(Node* root, SReal* minBBox, SReal* maxBBox);
/// Render the scene
void SOFA_SIMULATION_CORE_API draw(sofa::core::visual::VisualParams* vparams, Node* root);
/// Export a scene to an OBJ 3D Scene
void SOFA_SIMULATION_CORE_API exportOBJ(Node* root, const char* filename, bool exportMTL = true);
/// Print all objects in the graph in the given file (format is given by the filename extension)
void SOFA_SIMULATION_CORE_API exportGraph(Node* root, const char* filename=nullptr);
/// Dump the current state in the given stream
void SOFA_SIMULATION_CORE_API dumpState( Node* root, std::ofstream& out );
/// Load a scene from a file
NodeSPtr SOFA_SIMULATION_CORE_API load(const std::string& /* filename */, bool reload = false, const std::vector<std::string>& sceneArgs = std::vector<std::string>(0));
/// Unload a scene from a Node.
void SOFA_SIMULATION_CORE_API unload(NodeSPtr root);

}

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

    /// Print all object in the graph
    SOFA_ATTRIBUTE_DEPRECATED_SIMULATION_PRINT()
    virtual void print(Node* root);

    /// Initialize the objects
    SOFA_ATTRIBUTE_DEPRECATED_SIMULATION_INIT()
    virtual void init(Node* root);

    ///Init a node without changing the context of the simulation.
    SOFA_ATTRIBUTE_DEPRECATED_SIMULATION_INITNODE()
    virtual void initNode(Node* node);

    /// Execute one timestep. If dt is 0, the dt parameter in the graph will be used
    SOFA_ATTRIBUTE_DEPRECATED_SIMULATION_ANIMATE()
    virtual void animate(Node* root, SReal dt=0.0);

    /// Update the Visual Models: triggers the Mappings
    SOFA_ATTRIBUTE_DEPRECATED_SIMULATION_UPDATEVISUAL()
    virtual void updateVisual(Node* root);

    /// Reset to initial state
    SOFA_ATTRIBUTE_DEPRECATED_SIMULATION_RESET()
    virtual void reset(Node* root);

    /// Initialize the textures
    SOFA_ATTRIBUTE_DEPRECATED_SIMULATION_INITTEXTURE()
    virtual void initTextures(Node* root);

    /// Update contexts. Required before drawing the scene if root flags are modified.
    SOFA_ATTRIBUTE_DEPRECATED_SIMULATION_UPDATECONTEXT()
    virtual void updateContext(Node* root);

    /// Update contexts. Required before drawing the scene if root flags are modified.
    SOFA_ATTRIBUTE_DEPRECATED_SIMULATION_UPDATEVISUALCONTEXT()
    virtual void updateVisualContext(Node* root);

    /** Compute the bounding box of the scene.
     * If init is set to "true", then minBBox and maxBBox will be initialised to a default value
     * @warning MechanicalObjects with showObject member set to false are ignored
     * @sa computeTotalBBox(Node* root, SReal* minBBox, SReal* maxBBox)
     */
    SOFA_ATTRIBUTE_DEPRECATED_SIMULATION_COMPUTEBBOX()
    virtual void computeBBox(Node* root, SReal* minBBox, SReal* maxBBox, bool init=true);

    /** Compute the bounding box of the scene.
     * Includes all objects, may they be displayed or not.
     * @sa computeBBox(Node* root, SReal* minBBox, SReal* maxBBox, bool init=true)
     * @deprecated
     */
    SOFA_ATTRIBUTE_DEPRECATED_SIMULATION_COMPUTETOTALBBOX()
    virtual void computeTotalBBox(Node* root, SReal* minBBox, SReal* maxBBox);

    /// Render the scene
    SOFA_ATTRIBUTE_DEPRECATED_SIMULATION_DRAW()
    virtual void draw(sofa::core::visual::VisualParams* vparams, Node* root);

    /// Export a scene to an OBJ 3D Scene
    SOFA_ATTRIBUTE_DEPRECATED_SIMULATION_EXPORTOBJ()
    virtual void exportOBJ(Node* root, const char* filename, bool exportMTL = true);

    /// Print all object in the graph in XML format
    SOFA_ATTRIBUTE_DEPRECATED_SIMULATION_EXPORTXML()
    virtual void exportXML(Node* root, const char* fileName=nullptr);

    /// Print all objects in the graph in the given file (format is given by the filename extension)
    SOFA_ATTRIBUTE_DEPRECATED_SIMULATION_EXPORTGRAPH()
    virtual void exportGraph(Node* root, const char* filename=nullptr);

    /// Dump the current state in the given stream
    SOFA_ATTRIBUTE_DEPRECATED_SIMULATION_DUMPSTATE()
    virtual void dumpState( Node* root, std::ofstream& out );

    /// Load a scene from a file
    SOFA_ATTRIBUTE_DEPRECATED_SIMULATION_LOAD()
    virtual NodeSPtr load(const std::string& /* filename */, bool reload = false, const std::vector<std::string>& sceneArgs = std::vector<std::string>(0));

    /// Unload a scene from a Node.
    SOFA_ATTRIBUTE_DEPRECATED_SIMULATION_UNLOAD()
    virtual void unload(NodeSPtr root);

    /// create a new graph(or tree) and return its root node.
    virtual NodeSPtr createNewGraph(const std::string& name)=0;//Todo replace newNode method

    /// creates and returns a new node.
    virtual NodeSPtr createNewNode(const std::string& name)=0;

    /// Can the simulation handle a directed acyclic graph?
    virtual bool isDirectedAcyclicGraph() = 0;

    inline static Simulation::SPtr theSimulation { nullptr };
};
} // namespace sofa::simulation

MSG_REGISTER_CLASS(sofa::simulation::Simulation, "Simulation")
