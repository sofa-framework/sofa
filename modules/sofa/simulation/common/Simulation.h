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
#include <sofa/helper/gl/DrawManager.h>
#include <sofa/helper/gl/VisualParameters.h>
#include <sofa/core/componentmodel/behavior/BaseMechanicalMapping.h>
#include <sofa/core/objectmodel/DataFileName.h>

namespace sofa
{

namespace simulation
{


/** Main controller of the scene.
Defines how the scene is inited at the beginning, and updated at each time step.
Derives from BaseObject in order to model the parameters as Datas, which makes their edition easy in the GUI.
*/
class SOFA_SIMULATION_COMMON_API Simulation: public virtual sofa::core::objectmodel::BaseObject
{
public:

    Simulation();
    virtual ~Simulation();

    /// Print all object in the graph
    virtual void print(Node* root);

    /// Print all object in the graph in XML format
    virtual void printXML(Node* root, const char* fileName=0, bool compact=false);

    /// Initialize the objects
    virtual void init(Node* root);

    /// Find the list of nodes called "Instrument" and keep it in the vector instuments
    void getInstruments( Node *node);

    /// Execute one timestep. If dt is 0, the dt parameter in the graph will be used
    virtual void animate(Node* root, double dt=0.0);

    /// Reset to initial state
    virtual void reset(Node* root);

    /// Initialize the textures
    virtual void initTextures(Node* root);

    /// Update contexts. Required before drawing the scene if root flags are modified.
    virtual void updateContext(Node* root);

    /// Update contexts. Required before drawing the scene if root flags are modified.
    virtual void updateVisualContext(Node* root,int FILTER=10);

    /// Compute the bounding box of the scene.
    virtual void computeBBox(Node* root, SReal* minBBox, SReal* maxBBox);

    /// Render the scene
    virtual void draw(Node* root, helper::gl::VisualParameters* params = NULL);

    /// Render the scene - Shadows pass
    virtual void drawShadows(Node* root);

    /// Export a scene to an OBJ 3D Scene
    virtual void exportOBJ(Node* root, const char* filename, bool exportMTL = true);

    /// Export a scene to XML
    virtual void exportXML(Node* root, const char* filename);

    /// Dump the current state in the given stream
    virtual void dumpState( Node* root, std::ofstream& out );

    /// Initialize gnuplot export (open files)
    virtual void initGnuplot( Node* root );
    /// Dump the current state in gnuplot files
    virtual void exportGnuplot( Node* root, double time );

    /// Load a scene from a file.
    virtual Node* load(const char* /* filename */)=0;
    /// Unload a scene from a Node.
    virtual void unload(Node * /* root */)=0;


    /// Create a new Node of the simulation
    virtual Node* newNode(const std::string& name)=0;

    /// During init, the components will tell if they need prefetch phase. If only one needs it, we have to use it.
    void setPrefteching(bool b) { needToPrefetch |= b;};

    bool isPrefetchingUsed() const { return needToPrefetch;};

    /// Number of mechanical steps within an animation step
    Data<unsigned> numMechSteps;

    /// Number of steps of simulation
    unsigned nbSteps;

    /// Need to launch prefetches during animating steps
    bool needToPrefetch;

    sofa::core::objectmodel::DataFileName gnuplotDirectory;

    helper::vector< Node* > instruments;
    Data< int > instrumentInUse;

    sofa::helper::gl::DrawManager DrawUtility;
    static  Simulation* theSimulation;
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
