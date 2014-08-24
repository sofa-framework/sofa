#ifndef SOFA_NEWGUI_SofaGLScene_H
#define SOFA_NEWGUI_SofaGLScene_H


#include <SofaSimpleGUI/SofaGL.h>


using namespace sofa;
using simulation::Node;

namespace sofa {
namespace newgui {


/** @brief Simple Sofa interface to integrate in a graphics application with a single viewer.
 *
 * The API corresponds to the typical graphics callbacks: init, draw, animate, pick…
 * It basically puts together a SofaScene with a SofaGL.
 *
 * @author Francois Faure, 2014
 * */
class SOFA_SOFASIMPLEGUI_API SofaGLScene : public SofaScene
{

public:

    SofaGLScene();

    virtual ~SofaGLScene();

    /**
     * @brief Initialize Sofa and load a scene file
     * @param fileName Scene file to load
     */
    void init( const std::string& fileName="" );

    /**
     * @brief Initialize Sofa with a given graph
     */
    void init( Node::SPtr groot );

    // standard callback-level interface

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
    void attach( Interactor*  i );

    /**
     * @brief move the interactor according to the mouse pointer.
     * x,y in image coordinates (origin on top left).
     */
    void move( Interactor* i, int x, int y);

    /// Remove the interactor from the scene, without deleting it.
    void detach(Interactor* i);



protected:
    SofaGL *sofaGL;


};



}// newgui
}// sofa


#endif
