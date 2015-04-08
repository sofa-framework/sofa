#ifndef SOFA_NEWGUI_SofaGL_H
#define SOFA_NEWGUI_SofaGL_H

#include "initSimpleGUI.h"
#include <sofa/config.h>
#include "SofaScene.h"
#include <sofa/core/visual/DrawToolGL.h>

#include "PickedPoint.h"
#include "SpringInteractor.h"

namespace sofa {
namespace simplegui {

/** OpenGL interface to a SofaScene.
 * This is not a viewer, this is an object used by a viewer to display a Sofa scene and to pick objects in it.
 * It contains a pointer to the Sofa scene. Several viewers can be connected to a single scene through such interfaces.
 *
 * Picking returns a PickedPoint which describes a particle.
 * It is up to the application to create the appropriate Interactor, which can then be inserted in the Sofa scene.
 * This class provides the functions to attach/detach an interactor and move it.
 *
 * @author Francois Faure, 2014

 */
class SOFA_SOFASIMPLEGUI_API  SofaGL
{
public:
    /**
     * @param s The Sofa scene to interact with, the scene MUST already be opened !
     */
    SofaGL( SofaScene* s );
    /**
     * @brief Draw the scene and stores the transformation matrices, for picking.
     * This requires an OpenGL context. It is supposed to be used by the drawing method of a viewer, after setting the modelview matrix.
     */
    void draw();
    /**
     * @brief Compute the parameters to pass to gluPerspective to make the whole scene visible.
     * The new camera center is set on the line from the current camera center to the scene center, at the appropriate distance.
     * @param xcam Camera center (input-output)
     * @param ycam Camera center (input-output)
     * @param zcam Camera center (input-output)
     * @param xcen Center of the scene (output)
     * @param ycen Center of the scene (output)
     * @param zcen Center of the scene (output)
     * @param a Camera vertical angle (input)
     * @param near Smaller than the nearest distance from the new camera center to the scene (output)
     * @param far Larger than the nearest distance from the new camera center to the scene (output)
     */
    void viewAll( SReal* xcam, SReal* ycam, SReal* zcam, SReal* xcen, SReal* ycen, SReal* zcen, SReal a, SReal* nearPlane, SReal* farPlane);

    /// Compute the bounding box of the scene
    void getSceneBBox( float* xmin, float* ymin, float* zmin, float* xmax, float* ymax, float* zmax );

    /// @name Interaction
    /// @{
    /** @brief Try to pick a particle along a ray.
     * The ray starts at the camera center and passes through point with coordinates x,y
     * ox, oy, oz are the camera center in world coordinates.
     * x,y in image coordinates (origin on top left).
     * If a point is picked, the application may create an Interactor based on it.
     * @return a valid PickedPoint if succeeded, an invalid PickedPoint if not.
     */
    PickedPoint pick(GLdouble ox, GLdouble oy, GLdouble oz, int x, int y);


    /** @brief Insert an interactor in the scene
     * Does not check if it is already there, so be careful not to insert the same twice
     */
    void attach( Interactor*  );

    /**
     * @brief getInteractor
     * @param picked
     * @return Interactor acting on the given picked point, or NULL if none
     */
    Interactor* getInteractor( const PickedPoint& picked );


    /** @brief Try to pick an Interactor along a ray.
     * The ray starts at the camera center and passes through point with coordinates x,y
     * ox, oy, oz are the camera center in world coordinates.
     * x,y in image coordinates (origin on top left).
     * @return Pointer if an Interactor is found, NULL if not.
     */
    Interactor* pickInteractor(GLdouble ox, GLdouble oy, GLdouble oz, int x, int y);

    /**
     * @brief move the interactor according to the mouse pointer.
     * x,y in image coordinates (origin on top left).
     */
    void move( Interactor*, int x, int y);

    /// Remove the interactor from the scene, without deleting it.
    void detach(Interactor*);

    /// @}

    /** @brief Try to pick a displayed thing along a ray, using OpenGL picking.
     * The ray starts at the camera center and passes through point with coordinates x,y in image coordinates (origin on top left).
     * In this experimental implementation, this function just prints something on the standard output.
     */
    void glPick(int x, int y);

    /**
     * @brief getPickDirection Compute the direction of a button click, returned as a unit vector
     * @param dx normalized direction
     * @param dy normalized direction
     * @param dz normalized direction
     * @param x x-coordinate of the click
     * @param y y-coordinate of the click (origin on top)
     */
    void getPickDirection( GLdouble* dx, GLdouble* dy, GLdouble* dz, int x, int y );


protected:
    SofaScene* _sofaScene;

    // matrices used for picking
    GLint _viewport[4];
    GLdouble _mvmatrix[16], _projmatrix[16];

    // rendering tools
    sofa::core::visual::DrawToolGL   _drawToolGL;
    sofa::core::visual::VisualParams* _vparams;
    void draw(sofa::core::visual::VisualParams*);

    // Interaction tools
    typedef map< PickedPoint, Interactor*> Picked_to_Interactor;
    /** Currently available interactors, associated with picked points.
     *  The interactors are not necessarily being manipulated. Only one is typically manipulated at a given time.
     */
    Picked_to_Interactor _picked_to_interactor;
    Interactor* _drag;                            ///< The currently active interactor



    // OpenGL picking data
    bool _isPicking;        ///< if we are doing OpenGL picking
    GLint pickX, pickY;     // window coordinates of the pick point
    enum {BUFSIZE=1024};
    GLuint selectBuf[BUFSIZE];
    GLint hits;


};

}
}

#endif // SOFA_NEWGUI_SofaGL_H
