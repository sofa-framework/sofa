#include "Camera.h"
#include <GL/gl.h>
#include <iostream>
using std::cout;
using std::endl;

namespace sofa{
namespace simplegui{

Camera::Camera()
{
    viewMode = EXAMINER;
    transform.matrix() = Eigen::Matrix4f::Identity();
    tb_tournerXY=0, tb_translaterXY=0, tb_bougerZ=0;
}

void Camera::glMultViewMatrix()
{

        glMultMatrixf( transform.data() );
}

void Camera::lookAt( Vec3 eye, Vec3 target, Vec3 upVec )
{
//    cout<<"Camera::lookAt " << eye.transpose() <<", " << target.transpose() << ", " << upVec.transpose() << endl;

    Vec3 forward = target - eye;
    forward.normalize();

    Vec3 side = forward.cross(upVec);
    side.normalize();

    Vec3 up = side.cross(forward);
    // The column vectors (side,up,-forward,eye) define the pose of the camera in world coordinates
    // The desired transformation is the inverse of the latter.

    // Transpose of the camera orientation
    for(int i=0; i<3; i++){
        transform.linear()(0,i) = side(i);
        transform.linear()(1,i) = up(i);
        transform.linear()(2,i) = -forward(i);
     }

    // -orientation.transpose*translation
    transform.translation() = -transform.linear() * eye;

//    cout<<"  transform matrix: " << endl << transform.matrix() << endl;

}

bool Camera::handleMouseButton( int button, int state, int x, int y )
{
    if( button==ButtonLeft && state==ButtonDown )
    {
        tb_tournerXY = 1;
        tb_ancienX = x;
        tb_ancienY = y;
        return true;
    }
    else if( button==ButtonLeft && state==ButtonUp )
    {
        tb_tournerXY = 0;
        return true;
    }
    if( button==ButtonMiddle && state==ButtonDown )
    {
        tb_bougerZ = 1;
        tb_ancienX = x;
        tb_ancienY = y;
        return true;
    }
    else if( button==ButtonMiddle && state==ButtonUp )
    {
        tb_bougerZ = 0;
        return true;
    }
    else if( button==ButtonRight && state==ButtonDown )
    {
        tb_translaterXY = 1;
        tb_ancienX = x;
        tb_ancienY = y;
        return true;
    }
    else if( button==ButtonRight && state==ButtonUp )
    {
        tb_translaterXY = 0;
        return true;
    }
    return false;
}



bool Camera::handleMouseMotion( int x, int y )
{
    float dx,dy;

    if( tb_tournerXY || tb_translaterXY || tb_bougerZ )
    {
        dx = x - tb_ancienX;
        dy = tb_ancienY - y;

        if( tb_tournerXY )
        {
            float angle = sqrt(dx*dx+dy*dy)/100;
            Vec3 axis(-dy, dx, 0);
            if( axis.norm()<1.0e-6 ) return true;
            axis.normalize();
            Eigen::AngleAxisf rot( angle, axis );
            transform.linear() = rot * transform.linear();

        }
        else if( tb_translaterXY )
        {
            transform.translation() += Vec3( dx/100.0, dy/100, 0);
        }
        else if( fabs(dx)>fabs(dy) )
        { // rotation z
            float angle = dx;
            Eigen::AngleAxisf rot( angle, Vec3(0,0,-1) );
            transform.linear() = rot*transform.linear();
        }
        else if( fabs(dy)>fabs(dx) )
        {
            transform.translation() += Vec3( 0,0, -dy/100);
        }
        tb_ancienX = x;
        tb_ancienY = y;
        return true;
    }
    return false;
}

} // simplegui
} // sofa

