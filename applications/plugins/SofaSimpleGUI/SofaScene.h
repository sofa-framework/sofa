#ifndef SOFA_NEWGUI_SofaScene_H
#define SOFA_NEWGUI_SofaScene_H

#include "initSimpleGUI.h"
#include <sofa/config.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/simulation/graph/DAGSimulation.h>
typedef sofa::simulation::graph::DAGSimulation SofaSimulation;
//#include <sofa/simulation/tree/TreeSimulation.h>
//typedef sofa::simulation::tree::TreeSimulation SofaSimulation;

namespace sofa {
using simulation::Node;

namespace simplegui {

class Interactor;


/** @brief A sofa scene graph with simulation functions.
 *
 * The typical life cycle is:
 *
 * SofaScene();
 * loadPlugins( list of plugin names );
 * setScene( filename or scenegraph );
 * [ your main loop: ]
 *      step(dt);
 *      [ use a SofaGL object to display the simulated objects ]
 * ~SofaScene();
 *
 * Node _groot is the root of the scene.
 * Interactors are set under its child node _iroot.

 * @author Francois Faure, 2014
 */
class SOFA_SOFASIMPLEGUI_API  SofaScene : public SofaSimulation
{
protected:


    Node::SPtr _groot; ///< root of the scene
    Node::SPtr _iroot; ///< root of the interactors, child of _groot
    std::string _currentFileName; ///< Name of the current scene

public:
    /**
     * @brief Initialize Sofa
     */
    SofaScene();
    virtual ~SofaScene(){}


    /**
     * @brief load the given plugin
     * @param pluginName name of the plugin
     */
    void loadPlugins( std::vector<std::string> pluginName );
    /**
     * @brief Load a scene file. The previous scene graph, if any, is deleted.
     * @param fileName Scene file to load
     */
    void setScene( const std::string& fileName );
    /**
     * @brief Set the scene graph. The previous scene graph, if any, is deleted.
     * @param graph the scene to simulate
     */
    void setScene( Node::SPtr graph );
    /**
     * @brief Print the scene graph on the standard ouput, for debugging.
     */
    void printGraph();
    /**
     * @brief Integrate time by one step and update the Sofa scene.
     */
    void step( SReal dt );
    /**
     * @brief restart from the beginning
     */
    void reset();
    /**
     * @brief Compute the bounding box of the simulated objects
     * @param xmin min coordinate in the X direction
     * @param xmax max coordinate in the X direction
     * @param ymin etc.
     * @param ymax
     * @param zmin
     * @param zmax
     */
    void getBoundingBox( SReal* xmin, SReal* xmax, SReal* ymin, SReal* ymax, SReal* zmin, SReal* zmax );

    /**
     * @return The root of the loaded scene.
     */
    Node::SPtr groot(){ return _groot; }
    /**
     * @return The root of the interactors.
     */
    Node::SPtr iroot(){ return _iroot; }

    /// Do not use this directly. Use Interactor::attach, which calls this.
    void insertInteractor( Interactor* );


};

}
}


#endif // SOFA_NEWGUI_SofaScene_H
