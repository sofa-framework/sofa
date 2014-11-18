#ifndef SOFA_NEWGUI_SofaScene_H
#define SOFA_NEWGUI_SofaScene_H

#include "initSimpleGUI.h"
#include <sofa/config.h>
#include <vector>
#include <string>

namespace sofa {
namespace simulation {
    class Simulation;
    class Node;
}

namespace simplegui {

class Interactor;


/** @brief A sofa scene graph with simulation functions.
 *
 * There are methods to initialize and update the visual models, but rendering must be performed externally, see e.g. class SofaGL.
 *
 * The typical life cycle is:
 *
        loadPlugins( list of plugin names );
        setScene( scenegraph ); or open(filename)
        initVisual()
        [ your main loop: ]
            step(dt);
            updateVisual();
            [ use a SofaGL object to display the simulated objects ]


 * @author Francois Faure, 2014
 */
class SOFA_SOFASIMPLEGUI_API  SofaScene
{
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
    void open( const std::string& fileName );
    /**
     * @brief Set the scene graph. The previous scene graph, if any, is deleted.
     * @param graph the scene to simulate
     */
    void setScene( simulation::Node* graph );
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

    /// To do once before rendering a scene, typically at initialization time
    void initVisual();

    /// Update the visual models. To do after animating and before rendering.
    void updateVisual();



    /** @name Developer API
     * To be used to create new functionalities.
     */
    ///@{

    /// Root of the simulated scene.
    simulation::Node* groot();

    /// Root of the interactors, set as child of groot
    simulation::Node* iroot(){ return _iroot; }

    /// Do not use this directly. Use Interactor::attach, which calls this.
    void insertInteractor( Interactor* );

    ///@}




protected:
    simulation::Node* _groot; ///< root of the scene
    simulation::Node* _iroot; ///< root of the interactors, child of _groot
    simulation::Simulation* sofaSimulation;

};

}
}


#endif // SOFA_NEWGUI_SofaScene_H
