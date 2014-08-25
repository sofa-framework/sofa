#ifndef SOFA_SIMPLEGUI_CAMERA_H
#define SOFA_SIMPLEGUI_CAMERA_H

#include <Eigen/Geometry>

namespace sofa{
namespace simplegui{

/**
 * @brief The Camera class implements a simple viewpoint transformation, and its update using the mouse.
 * Currently only one displacement mode is implemented, and it is not extensively tested.
 *
 * @author Francois Faure, 2014
 */
class Camera
{
public:
    typedef Eigen::Transform<float,3,Eigen::Affine,Eigen::ColMajor> Transform;
    typedef Eigen::Vector3f Vec3;

    typedef enum {EXAMINER} ViewMode; // currently only one mode
    enum {ButtonLeft, ButtonMiddle, ButtonRight};
    enum {ButtonDown,ButtonUp};


    Camera();
    void setViewMode( ViewMode ){} // currently does nothing


    /// Apply the viewing transform, typically just after glLoadIdentity() in the draw function.
    void glMultViewMatrix();

    /// Equivalent of gluLookAt.
    void lookAt( Vec3 eye, Vec3 target, Vec3 up );
    /// Set the camera displacement modes and return true.
    bool handleMouseButton(int button, int state, int x, int y);
    /// Displace the camera based on the mouse motion and return true.
    bool handleMouseMotion(int x, int y);

protected:

    ViewMode viewMode; // not used yet
    Transform transform; ///< Viewing transform: world wrt camera, i.e. inverse of the camera pose


    int tb_ancienX, tb_ancienY, tb_tournerXY, tb_translaterXY, tb_bougerZ;

};



}
}


#endif // CAMERA_H
