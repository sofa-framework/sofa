/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_SIMULATION_CORE_SIMULATION_H
#define SOFA_SIMULATION_CORE_SIMULATION_H

#include <sofa/simulation/Node.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/visual/DisplayFlags.h>
#include <memory>

#include <sofa/simulation/DefaultAnimationLoop.h>
#include <sofa/simulation/DefaultVisualManagerLoop.h>

namespace sofa
{

namespace simulation
{


/** Main controller of the scene.
    Defines how the scene is inited at the beginning, and updated at each time step.
    Derives from Base in order to use smart pointers and model the parameters as Datas, which makes their edition easy in the GUI.
 */
class SOFA_SIMULATION_CORE_API Simulation: public virtual sofa::core::objectmodel::Base
{
public:
    SOFA_CLASS(Simulation, sofa::core::objectmodel::Base);

    typedef sofa::core::visual::DisplayFlags DisplayFlags;
// protected:
    Simulation();
    virtual ~Simulation();
	
private:
	Simulation(const Simulation& n) ;
	Simulation& operator=(const Simulation& n) ;
	
public:
    /// Print all object in the graph
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
    virtual void exportXML(Node* root, const char* fileName=NULL);

    /// Print all objects in the graph in the given file (format is given by the filename extension)
    virtual void exportGraph(Node* root, const char* filename=NULL);

    /// Dump the current state in the given stream
    virtual void dumpState( Node* root, std::ofstream& out );

    /// Load a scene from a file
    virtual Node::SPtr load(const char* /* filename */);

    /// Unload a scene from a Node.
    virtual void unload(Node::SPtr root);

    /// create a new graph(or tree) and return its root node.
    virtual Node::SPtr createNewGraph(const std::string& name)=0;//Todo replace newNode method

    /// creates and returns a new node.
    virtual Node::SPtr createNewNode(const std::string& name)=0;

    /// @warning this singleton has one limitation: it is easy to create several types of
    /// simulations at the same time (e.g. DAGSimulation + TreeSimulation)
    /// but it does not sound like a huge limitation
    static Simulation::SPtr theSimulation;

    /// Can the simulation handle a directed acyclic graph?
    virtual bool isDirectedAcyclicGraph() = 0;

};

/// Set the (unique) simulation which controls the scene
SOFA_SIMULATION_CORE_API void setSimulation(Simulation* s);

/** Get the (unique) simulation which controls the scene.
    Automatically creates one if no Simulation has been set.
 */
SOFA_SIMULATION_CORE_API Simulation* getSimulation();

} // namespace simulation

} // namespace sofa

#endif
