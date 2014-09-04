#include "Camera.h"
#include <GL/glew.h>
#include <iostream>
using std::cout;
using std::endl;

namespace sofa{
namespace simplegui{

Camera::Camera()
{
    transform.matrix() = Eigen::Matrix4f::Identity();
    tb_tournerXY=0, tb_translaterXY=0, tb_bougerZ=0;
}

void Camera::lookAt()
{

    glMultMatrixf( transform.data() );
}

template <typename T> inline T sqr(const T& t){ return t*t; }

void Camera::viewAll( float xmin, float ymin, float zmin, float xmax, float ymax, float zmax )
{
    Vec3 pmin(xmin,ymin,zmin), pmax(xmax,ymax,zmax);
    Vec3 pcen = (pmin+pmax)*0.5;
    Vec3 diag = pmax-pmin;
    float radius = diag.norm();
    //    cout<<"Camera, diag = " << diag.transpose() << endl;
    //    cout<<"Camera, scene radius = " << radius << endl;

    // Desired distance:  distance * tan(a) = radius
    float distance = 1.5 * radius / tan(fovy * 3.1415927/180);
    //    cout<<"Camera::viewAll, angle = " << fovy << ", tan = " << tan(fovy) << ", distance = " << distance << endl;
    //    cout<<"Camera::viewAll, xmin xmax ymin ymax zmin zmax = " << xmin << " " << xmax <<" "<<ymin<<" "<<ymax<<" "<<zmin<<" "<<zmax<< endl;

    // move the camera along the current camera-center line, at the right distance
    // cam = cen + distance * (cam-cen)/|cam-cen|
    Vec3 forward = pcen - eye();
    float curdist = forward.norm();
    Vec3 peye = pcen - forward * distance / curdist;

    // update the depth bounds
    znear = distance - radius*1.5;
    zfar  = distance + radius*1.5;

    setlookAt(peye(0),peye(1),peye(2),
              pcen(0),pcen(1),pcen(2),
              transform.linear()(1,0), transform.linear()(1,1), transform.linear()(1,2)); // use current y direction as up axis
}


void Camera::perspective( float f, float r, float zn, float zf )
{
    setPerspective(f,r,zn,zf);
    perspective();
}

void Camera::setPerspective( float f, float r, float zn, float zf )
{
    fovy=f, ratio=r, znear=zn, zfar=zf;
}

void Camera::perspective()
{
    gluPerspective(fovy,ratio,znear,zfar);
}


void Camera::setlookAt(
        float eyeX, float eyeY, float eyeZ,
        float targetX, float targetY, float targetZ,
        float upX, float upY, float upZ
        )
{
    Vec3 eye(eyeX,eyeY,eyeZ), target(targetX,targetY,targetZ), upVec(upX,upY,upZ);
    //    cout<<"Camera::setLookAt " << eye.transpose() <<", " << target.transpose() << ", " << upVec.transpose() << endl;

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

    //    cout<<"Camera::setLookAt,  transform matrix: " << endl << transform.matrix() << endl;

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
        else {
//            if( fabs(dx)>fabs(dy) )
//            { // rotation z
//                float angle = dx;
//                Eigen::AngleAxisf rot( angle, Vec3(0,0,-1) );
//                transform.linear() = rot*transform.linear();
//            }
//            else if( fabs(dy)>fabs(dx) )
//            {
                transform.translation() += Vec3( 0,0, -dy/100);
//            }
        }
        tb_ancienX = x;
        tb_ancienY = y;
        return true;
    }
    return false;
}

Camera::Vec3 Camera::eye() const {
    return - transform.linear().inverse() * transform.translation();
}


} // simplegui
} // sofa

