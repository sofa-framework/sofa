#ifndef SOFA_COMPONENTS_GRAPH_SIMULATION_H
#define SOFA_COMPONENTS_GRAPH_SIMULATION_H

#include "GNode.h"

namespace Sofa
{

namespace Components
{

namespace Graph
{

class Simulation
{
public:

    /// Load a scene from a file
    static GNode* load(const char* filename);

    /// Print all object in the graph
    static void print(GNode* root);

    /// Initialize the objects
    //static void init(GNode* root);

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

};

} // namespace Graph

} // namespace Components

} // namespace Sofa

#endif
