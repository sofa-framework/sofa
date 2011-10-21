/******************************************************************************
 *       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
 *                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_SIMULATION_COMMON_SIMULATION_H
#define SOFA_SIMULATION_COMMON_SIMULATION_H

#include <sofa/simulation/common/Node.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/visual/DisplayFlags.h>
#include <sofa/simulation/common/xml/BaseElement.h>
#include <sofa/simulation/common/xml/XML.h>
#include <memory>

#include <sofa/simulation/common/DefaultAnimationLoop.h>
#include <sofa/simulation/common/DefaultVisualManagerLoop.h>

namespace sofa
{

namespace simulation
{


/** Main controller of the scene.
    Defines how the scene is inited at the beginning, and updated at each time step.
    Derives from Base in order to use smart pointers and model the parameters as Datas, which makes their edition easy in the GUI.
 */
class SOFA_SIMULATION_COMMON_API Simulation: public virtual sofa::core::objectmodel::Base
{
public:
    SOFA_CLASS(Simulation, sofa::core::objectmodel::Base);

    typedef sofa::core::visual::DisplayFlags DisplayFlags;
// protected:
    Simulation();
    virtual ~Simulation();
public:
    /// Print all object in the graph
    virtual void print(Node* root);

    /// Initialize the objects
    virtual void init(Node* root);

    ///Init a node without changing the context of the simulation.
    virtual void initNode(Node* node);


    /// Execute one timestep. If dt is 0, the dt parameter in the graph will be used
    virtual void animate(Node* root, double dt=0.0);

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

    /// Compute the bounding box of the scene. If init is set to "true", then minBBox and maxBBox will be initialised to a default value
    virtual void computeBBox(Node* root, SReal* minBBox, SReal* maxBBox, bool init=true);

    /// Render the scene
    virtual void draw(sofa::core::visual::VisualParams* vparams, Node* root);

    /// Export a scene to an OBJ 3D Scene
    virtual void exportOBJ(Node* root, const char* filename, bool exportMTL = true);

    /// Print all object in the graph in XML format
    virtual void exportXML(Node* root, const char* fileName=NULL);

    /// Dump the current state in the given stream
    virtual void dumpState( Node* root, std::ofstream& out );

    /// Load a scene from a file.
    virtual Node::SPtr load(const char* /* filename */);
    /// Unload a scene from a Node.
    virtual void unload(Node::SPtr root);

    ///create a new graph(or tree) and return its root node
    virtual Node::SPtr createNewGraph(const std::string& name)=0;//Todo replace newNode method

    ///load a scene from memory (typically : an xml into a string)
    static Node::SPtr loadFromMemory ( const char *filename, const char *data, unsigned int size );
    ///load a scene from a file
    static Node::SPtr loadFromFile ( const char *filename );
    ///generic function to process xml tree (after loading the xml structure from the 2 previous functions)
    static Node::SPtr processXML(xml::BaseElement* xml, const char *filename);

    static Simulation::SPtr theSimulation;
};

/// Set the (unique) simulation which controls the scene
SOFA_SIMULATION_COMMON_API void setSimulation(Simulation* s);

/** Get the (unique) simulation which controls the scene.
    Automatically creates one if no Simulation has been set.
 */
SOFA_SIMULATION_COMMON_API Simulation* getSimulation();

} // namespace simulation

} // namespace sofa

#endif
