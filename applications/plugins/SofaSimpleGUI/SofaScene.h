#ifndef SOFA_NEWGUI_SofaScene_H
#define SOFA_NEWGUI_SofaScene_H

#include "initSimpleGUI.h"
#include <sofa/config.h>
#include <sofa/component/container/MechanicalObject.h>
#include <sofa/simulation/graph/DAGSimulation.h>
typedef sofa::simulation::graph::DAGSimulation SofaSimulation;
//#include <sofa/simulation/tree/TreeSimulation.h>
//typedef sofa::simulation::tree::TreeSimulation SofaSimulation;

namespace sofa {
using simulation::Node;

namespace newgui {

class Interactor;


/** @brief A sofa scene graph with simulation functions.
 * Node _groot is the root of the scene.
 * Interactors are set under its child node _iroot.
 *
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
     * @brief Initialize Sofa and create an empty scene graph.
     * The plugins are loaded later by the init function.
     */
    SofaScene();
    virtual ~SofaScene(){}
	/**
     * @return The root of the loaded scene.
     */
    Node::SPtr groot(){ return _groot; }
    /**
     * @return The root of the interactors.
     */
    Node::SPtr iroot(){ return _iroot; }
    /**
     * @brief Print the scene graph on the standard ouput, for debugging.
     */
    void printGraph();
    /**
     * @brief Initialize Sofa and load a scene file
     * @param plugins List of plugins to load
     * @param fileName Scene file to load
     */
    void init( std::vector<std::string> plugins, const std::string& fileName="" );
    /**
     * @brief Integrate time by one step and update the Sofa scene.
     */
    void step( SReal dt );
    /**
     * @brief restart from the beginning
     */
    void reset();
    /**
     * @brief Clear the current scene and load the given one
     * @param filename Scene description file
     */
    void open( const char* filename );
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

    /// Do not use this directly. Use Interactor::attach, which calls this.
    void insertInteractor( Interactor* );
};

}
}


#endif // SOFA_NEWGUI_SofaScene_H
