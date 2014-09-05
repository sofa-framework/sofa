#ifndef SOFA_SIMPLEGUI_CAMERA_H
#define SOFA_SIMPLEGUI_CAMERA_H

#include "initSimpleGUI.h"
#include <Eigen/Geometry>

namespace sofa{
namespace simplegui{

/**
 * @brief The Camera class implements a simple viewpoint transformation, and its update using the mouse.
 * Currently only one displacement mode is implemented, and it is not extensively tested.
 *
 * @author Francois Faure, 2014
 */
class SOFA_SOFASIMPLEGUI_API Camera
{
public:
    typedef Eigen::Transform<float,3,Eigen::Affine,Eigen::ColMajor> Transform;
    typedef Eigen::Vector3f Vec3;

    enum {ButtonLeft, ButtonMiddle, ButtonRight};
    enum {ButtonDown,ButtonUp};


    Camera();


    /// Set the view point. Parameters correspond to gluLookAt.
    /// Note that this just sets the parameters. The actual viewing transform is applied in void lookAt() .
    void setlookAt(
            float eyeX, float eyeY, float eyeZ,
            float targetX, float targetY, float targetZ,
            float upX, float upY, float upZ
            );

    /// Apply the viewing transform, typically just after glLoadIdentity() in the draw function.
    void lookAt();

    void viewAll( float xmin, float ymin, float zmin, float xmax, float ymax, float zmax );

    /// Equivalent of gluPerspective.
    /// Alternatively, the definition and the application of the projection matrix can be done separately using other functions.
    /// @sa void setPerspective( float fovy, float ratio, float znear, float zfar ) @sa void perspective()
    void perspective( float fovy, float ratio, float znear, float zfar );

    /// Set the projection matrix, without applying it. Parameters correspond to  gluPerspective.
    /// Note that this just sets the parameters. The actual projection transform is applied in void perspective() .
    void setPerspective( float fovy, float ratio, float znear, float zfar );

    /// Apply the projection matrix defined in setPerspective, typically in the reshape function
    void perspective();




    /// Set the camera displacement modes and return true.
    bool handleMouseButton(int button, int state, int x, int y);
    /// Displace the camera based on the mouse motion and return true.
    bool handleMouseMotion(int x, int y);

    /// Center of the camera in world coordinates
    Vec3 eye() const;

    /// Viewing matrix
    const Transform& getTransform() const { return transform; }

protected:

    Transform transform; ///< Viewing transform: world wrt camera, i.e. inverse of the camera pose

    // camera projection
    float fovy, ratio, znear, zfar;  // parameters of perspective


    int tb_ancienX, tb_ancienY, tb_tournerXY, tb_translaterXY, tb_bougerZ;

};



}
}


#endif // CAMERA_H
