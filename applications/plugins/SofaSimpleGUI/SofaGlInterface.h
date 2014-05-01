#ifndef SOFA_NEWGUI_SofaGlInterface_H
#define SOFA_NEWGUI_SofaGlInterface_H

#include <sofa/config.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/core/visual/DrawToolGL.h>

#include "PickedPoint.h"
#include "SpringInteractor.h"


using namespace sofa;
using simulation::Node;
typedef sofa::simulation::graph::DAGSimulation ParentSimulation;
//typedef sofa::simulation::tree::TreeSimulation ParentSimulation;

namespace sofa {
namespace newgui {


// sofa types should not be exposed
typedef sofa::defaulttype::Vector3 Vec3;
typedef sofa::component::container::MechanicalObject< defaulttype::Vec3Types > Vec3DOF;





/** @brief Simple Sofa interface to integrate in a graphics application.
 *
 * The API corresponds to the typical graphics callbacks: init, draw, animate, pick…
 * Picking returns a PickedPoint which describes a particle.
 * It is up to the application to create the appropriate Interactor, which can then be inserted in the Sofa scene.
 *
 * @author Francois Faure, 2014
 * */
class SofaGlInterface : public ParentSimulation
{
protected:

    // sofa types should not be exposed
    typedef sofa::defaulttype::Vector3 Vec3;
    typedef sofa::component::container::MechanicalObject< defaulttype::Vec3Types > Vec3DOF;

    Node::SPtr groot; ///< root of the graph
    Node::SPtr sroot; ///< root of the scene, child of groot


    GLint viewport[4];
    GLdouble mvmatrix[16], projmatrix[16];

    sofa::core::visual::DrawToolGL   drawToolGL;
    sofa::core::visual::VisualParams* vparams;
    Node::SPtr getRoot();


public:

    SofaGlInterface();

    // standard callback-level interface

    /**
     * @brief Initialize Sofa and load a scene file
     * @param plugins List of plugins to load
     * @param fileName Scene file to load
     */
    void init( /*std::vector<std::string>& plugins,*/ const std::string& fileName );

    std::vector<std::string> plugins; ///< list of plugins to load

    /**
     * @brief glDraw Draw the Sofa scene using OpenGL.
     * Requires that an OpenGL context is active.
     */
    void glDraw();

    /**
     * @brief Integrate time by one step and update the Sofa scene.
     */
    void animate();


    // user interaction

    /**
     * @brief Try to pick a particle.
     * ox, oy, oz are the camera center in world coordinates.
     * x,y in image coordinates (origin on top left).
     * If a point is picked, the application may create an Interactor based on it.
     * @return a valid PickedPoint if succeeded.
     */
    PickedPoint pick( GLdouble ox, GLdouble oy, GLdouble oz, int x, int y );

    /// Insert the interactor in the scene
    void attach( Interactor*  );

    /**
     * @brief move the interactor according to the mouse pointer.
     * x,y in image coordinates (origin on top left).
     */
    void move( Interactor*, int x, int y);

    /// Remove the interactor from the scene, without deleting it.
    void detach(Interactor*);


    // debugging

    bool debug;

    void printScene();

};



}// newgui
}// sofa


#endif
