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

#include <sofa/simulation/tree/GNode.h>

namespace sofa
{

namespace simulation
{

namespace tree
{

class Simulation
{
public:

    /// Load a scene from a file
    static GNode* load(const char* filename);

    /// Print all object in the graph
    static void print(GNode* root);

    /// Print all object in the graph in XML format
    static void printXML(GNode* root, const char* fileName=0);

    /// Initialize the objects
    static void init(GNode* root);

    /// Execute one timestep. If dt is 0, the dt parameter in the graph will be used
    static void animate(GNode* root, double dt=0.0);

    /// Reset to initial state
    static void reset(GNode* root);

    /// Initialize the textures
    static void initTextures(GNode* root);

    /// Update contexts. Required before drawing the scene if root flags are modified.
    static void updateContext(GNode* root);

    /// Compute the bounding box of the scene.
    static void computeBBox(GNode* root, double* minBBox, double* maxBBox);

    /// Render the scene
    static void draw(GNode* root);

    /// Render the scene - Shadows pass
    static void drawShadows(GNode* root);

    /// Delete a scene from memory. After this call the pointer is invalid
    static void unload(GNode* root);

    /// Export a scene to an OBJ 3D Scene
    static void exportOBJ(GNode* root, const char* filename, bool exportMTL = true);

    /// Export a scene to XML
    static void exportXML(GNode* root, const char* filename);

    /// Dump the current state in the given stream
    static void dumpState( GNode* root, std::ofstream& out );

    /// Initialize gnuplot export (open files)
    static void initGnuplot( GNode* root );
    /// Dump the current state in gnuplot files
    static void exportGnuplot( GNode* root, double time );

};

} // namespace tree

} // namespace simulation

} // namespace sofa

#endif
