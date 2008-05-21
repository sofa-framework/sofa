/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_SIMULATION_TREE_SIMULATION_H
#define SOFA_SIMULATION_TREE_SIMULATION_H

#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/tree/GNode.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/LaparoscopicRigidTypes.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/simulation/tree/xml/XML.h>
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
class Simulation: public virtual sofa::core::objectmodel::BaseObject
{
private:
    ///load a scene from memory (typically : an xml into a string)
    static GNode* loadFromMemory ( const char *filename, const char *data, unsigned int size );
    ///load a scene from a file
    static GNode* loadFromFile ( const char *filename );
    ///generic function to process xml tree (after loading the xml structure from the 2 previous functions)
    static GNode* processXML(xml::BaseElement* xml, const char *filename);
public:
    /** Load a scene from a file.
    Static method because at this point, the Simulation component is not yet created.
    If a Simulation component is found in the graph, then it is used.
    Otherwise, a default Simulation will be created at the first call to method getSimulation()
    This file can be a xml file or a script file which will generate a xml tree.
    */
    static GNode* load(const char* filename);

    Simulation();
    virtual ~Simulation();

    /// Print all object in the graph
    virtual void print(Node* root);

    /// Print all object in the graph in XML format
    virtual void printXML(Node* root, const char* fileName=0);

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
    virtual void updateVisualContext(Node* root,int FILTER=0);

    /// Compute the bounding box of the scene.
    virtual void computeBBox(Node* root, SReal* minBBox, SReal* maxBBox);

    /// Render the scene
    virtual void draw(Node* root);

    /// Render the scene - Shadows pass
    virtual void drawShadows(Node* root);

    /// Delete a scene from memory. After this call the pointer is invalid
    virtual void unload(GNode* root);

    /// Export a scene to an OBJ 3D Scene
    virtual void exportOBJ(GNode* root, const char* filename, bool exportMTL = true);

    /// Export a scene to XML
    virtual void exportXML(Node* root, const char* filename);

    /// Dump the current state in the given stream
    virtual void dumpState( Node* root, std::ofstream& out );

    /// Initialize gnuplot export (open files)
    virtual void initGnuplot( GNode* root );
    /// Dump the current state in gnuplot files
    virtual void exportGnuplot( GNode* root, double time );

    /// Number of mechanical steps within an animation step
    Data<unsigned> numMechSteps;
    Data<std::string> gnuplotDirectory;

    helper::vector< Node* > instruments;
    Data< int > instrumentInUse;
};

/// Set the (unique) simulation which controls the scene
void setSimulation(Simulation*);

/** Get the (unique) simulation which controls the scene.
Automatically creates one if no Simulation has been set.
*/
Simulation* getSimulation();

} // namespace tree

} // namespace simulation

} // namespace sofa

#endif
