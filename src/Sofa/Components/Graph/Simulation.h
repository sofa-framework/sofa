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

    /// Execute one timestep. If dt is 0, the dt parameter in the graph will be used
    static void animate(GNode* root, double dt=0.0);

    /// Reset to initial state
    static void reset(GNode* root);

    /// Initialize the textures
    static void initTextures(GNode* root);

    /// Render the scene
    static void draw(GNode* root);

};

} // namespace Graph

} // namespace Components

} // namespace Sofa

#endif
