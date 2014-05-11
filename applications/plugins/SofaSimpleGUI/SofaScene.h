#ifndef SOFA_NEWGUI_SofaScene_H
#define SOFA_NEWGUI_SofaScene_H

#include "initSimpleGUI.h"
#include <sofa/config.h>
#include <sofa/component/container/MechanicalObject.h>
#include <sofa/simulation/graph/DAGSimulation.h>
typedef sofa::simulation::graph::DAGSimulation SofaSimulation;

namespace sofa {
using simulation::Node;

namespace newgui {


/** @brief A sofa scene graph with simulation functions.
 * Node _groot is the root of the scene. Scene files are loaded under its child node _sroot.
 *
 * @author Francois Faure, 2014
 */
class SOFA_SOFASIMPLEGUI_API  SofaScene : public SofaSimulation
{
protected:

    // sofa types should not be exposed
    typedef sofa::defaulttype::Vector3 Vec3;
    typedef sofa::component::container::MechanicalObject< defaulttype::Vec3Types > Vec3DOF;

    Node::SPtr _groot; ///< root of the graph
    std::string _currentFileName; ///< Name of the current scene

public:
    /**
     * @brief Initialize Sofa and create an empty scene graph.
     * The plugins are loaded later by the init function.
     */
    SofaScene();
    virtual ~SofaScene(){}
    /**
     * @return the root of the scene graph
     */
    Node::SPtr groot();
    /**
     * @brief Print the scene graph on the standard ouput, for debugging.
     */
    void printScene();
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
};

}
}


#endif // SOFA_NEWGUI_SofaScene_H
