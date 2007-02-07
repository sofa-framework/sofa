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

    /// Delete a scene from memory. After this call the pointer is invalid
    static void unload(GNode* root);

    /// Export a scene to an OBJ 3D Scene
    static void exportOBJ(GNode* root, const char* filename, bool exportMTL = true);

    /// Export a scene to XML
    static void exportXML(GNode* root, const char* filename);

    /// Dump the current state in the given stream
    static void dumpState( GNode* root, std::ofstream& out );

};

} // namespace tree

} // namespace simulation

} // namespace sofa

#endif
