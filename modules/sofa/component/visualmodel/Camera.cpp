/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in theHope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You shouldHave received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L.Heigeas, C. Mendoza,   *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
/*
 * Camera.cpp
 *
 *      Author: froy
 */

#include <sofa/component/visualmodel/Camera.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/SolidTypes.h>
#include <sofa/helper/gl/Axis.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

SOFA_DECL_CLASS(Camera)

int CameraClass = core::RegisterObject("Camera")
        .add< Camera >()
        ;

Camera::Camera()
    :p_position(initData(&p_position, "position", "Camera's position"))
    ,p_orientation(initData(&p_orientation, "orientation", "Camera's orientation"))
    ,p_lookAt(initData(&p_lookAt, "lookAt", "Camera's look at"))
    ,p_distance(initData(&p_distance, "distance", "Distance between camera and look at"))
    ,p_fieldOfView(initData(&p_fieldOfView, (double) 45.0 , "fieldOfView", "Camera's FOV"))
    ,p_zNear(initData(&p_zNear, (double) 0.1 , "zNear", "Camera's zNear"))
    ,p_zFar(initData(&p_zFar, (double) 1000.0 , "zFar", "Camera's zFar"))
    ,p_minBBox(initData(&p_minBBox, Vec3(0.0,0.0,0.0) , "minBBox", "minBBox"))
    ,p_maxBBox(initData(&p_maxBBox, Vec3(1.0,1.0,1.0) , "maxBBox", "maaxBBox"))
    ,p_widthViewport(initData(&p_widthViewport, (unsigned int) 800 , "widthViewport", "widthViewport"))
    ,p_heightViewport(initData(&p_heightViewport,(unsigned int) 600 , "heightViewport", "heightViewport"))
    ,p_type(initData(&p_type, (int) Camera::PERSPECTIVE_TYPE, "type", "Camera Type (0 = Perspective, 1 = Orthographic)"))
    ,p_zoomSpeed(initData(&p_zoomSpeed, (double) 250.0 , "zoomSpeed", "Zoom Speed"))
    ,p_panSpeed(initData(&p_panSpeed, (double) 0.1 , "panSpeed", "Pan Speed"))
    ,currentMode(Camera::NONE_MODE)
    ,isMoving(false)
{

}

Camera::~Camera()
{

}
Camera::Vec3 Camera::cameraToWorldCoordinates(const Vec3& p)
{
    return p_orientation.getValue().rotate(p) + p_position.getValue();
}

Camera::Vec3 Camera::worldToCameraCoordinates(const Vec3& p)
{
    return p_orientation.getValue().inverseRotate(p - p_position.getValue());
}

Camera::Vec3 Camera::cameraToWorldTransform(const Vec3& v)
{
    Quat q = p_orientation.getValue();
    return q.rotate(v) ;
}

Camera::Vec3 Camera::worldToCameraTransform(const Vec3& v)
{
    return p_orientation.getValue().inverseRotate(v);
}

void Camera::getOpenGLMatrix(double mat[16])
{
    defaulttype::SolidTypes<double>::Transform world_H_cam(p_position.getValue(), this->getOrientation());
    world_H_cam.inversed().writeOpenGlMatrix(mat);
}

void Camera::init()
{
    currentTrackball.ComputeQuaternion(0.0, 0.0, 0.0, 0.0);

    if(p_position.isSet())
    {
        if(!p_orientation.isSet())
        {
            p_distance.setValue((p_lookAt.getValue() - p_position.getValue()).norm());

            Quat q  = getOrientationFromLookAt(p_position.getValue(), p_lookAt.getValue());
            p_orientation.setValue(q);
        }
        else if(!p_lookAt.isSet())
        {
            //distance assumed to be set
            if(!p_distance.isSet())
                sout << "Missing distance parameter ; taking default value (0.0, 0.0, 0.0)" << sendl;

            Vec3 lookat = getLookAtFromOrientation(p_position.getValue(), p_distance.getValue(), p_orientation.getValue());
            p_lookAt.setValue(lookat);
        }
        else
        {
            serr << "Too many missing parameters ; taking default ..." << sendl;
        }
    }
    else
    {
        if(p_lookAt.isSet() && p_orientation.isSet())
        {
            //distance assumed to be set
            if(!p_distance.isSet())
                sout << "Missing distance parameter ; taking default value (0.0, 0.0, 0.0)" << sendl;

            Vec3 pos = getPositionFromOrientation(p_lookAt.getValue(), p_distance.getValue(), p_orientation.getValue());
            p_position.setValue(pos);
        }
        else
        {
            serr << "Too many missing parameters ; taking default ..." << sendl;
        }
    }

    currentLookAt = p_lookAt.getValue();
    currentDistance = p_distance.getValue();

}

void Camera::reinit()
{
    //Data "LookAt" has changed
    //-> Orientation needs to be updated
    if(currentLookAt !=  p_lookAt.getValue())
    {
        Quat newOrientation = getOrientationFromLookAt(p_position.getValue(), p_lookAt.getValue());
        p_orientation.setValue(newOrientation);

        currentLookAt = p_lookAt.getValue();
    }
}

Camera::Quat Camera::getOrientationFromLookAt(const Camera::Vec3 &pos, const Camera::Vec3& lookat)
{
    Vec3 zAxis = -(lookat - pos);
    zAxis.normalize();

    Vec3 yAxis = cameraToWorldTransform(Vec3(0,1,0));

    Vec3 xAxis = yAxis.cross(zAxis) ;
    xAxis.normalize();

    //std::cout << xAxis.norm2() << std::endl;
    if (xAxis.norm2() < 0.00001)
        xAxis = cameraToWorldTransform(Vec3(1.0, 0.0, 0.0));
    xAxis.normalize();

    yAxis = zAxis.cross(xAxis);

    Quat q;
    q = q.createQuaterFromFrame(xAxis, yAxis, zAxis);
    q.normalize();
    return q;
}


Camera::Vec3 Camera::getLookAtFromOrientation(const Camera::Vec3 &pos, const double &distance, const Camera::Quat & orientation)
{
    Vec3 zWorld = orientation.rotate(Vec3(0,0,-1*distance));
    return zWorld+pos;
}

Camera::Vec3 Camera::getPositionFromOrientation(const Camera::Vec3 &lookAt, const double &distance, const Camera::Quat& orientation)
{
    Vec3 zWorld = orientation.rotate(Vec3(0,0,-1*distance));
    return zWorld-lookAt;
}

void Camera::rotateCameraAroundPoint(Quat& rotation, const Vec3& point)
{
    Vec3 tempAxis;
    double tempAngle;
    Quat orientation = this->getOrientation();
    Vec3& position = *p_position.beginEdit();
    double distance = (point - p_position.getValue()).norm();

    rotation.quatToAxis(tempAxis, tempAngle);
    //std::cout << tempAxis << " " << tempAngle << std::endl;
    Quat tempQuat (orientation.inverse().rotate(-tempAxis ), tempAngle);
    orientation = orientation*tempQuat;

    Vec3 trans = point + orientation.rotate(Vec3(0,0,-distance)) - position;
    position = position + trans;

    p_orientation.setValue(orientation);
    p_position.endEdit();
}

void Camera::rotateWorldAroundPoint(Quat& rotation, const Vec3&  point )
{
    Vec3 tempAxis;
    double tempAngle;
    Quat orientationCam = this->getOrientation();
    Vec3& positionCam = *p_position.beginEdit();

    rotation.quatToAxis(tempAxis, tempAngle);
    Quat tempQuat (orientationCam.rotate(-tempAxis), tempAngle);

    defaulttype::SolidTypes<double>::Transform world_H_cam(positionCam, orientationCam);
    defaulttype::SolidTypes<double>::Transform world_H_pivot(point, Quat());
    defaulttype::SolidTypes<double>::Transform pivotBefore_R_pivotAfter(Vec3(0.0,0.0,0.0), tempQuat);
    defaulttype::SolidTypes<double>::Transform camera_H_WorldAfter = world_H_cam.inversed() * world_H_pivot * pivotBefore_R_pivotAfter * world_H_pivot.inversed();
    //defaulttype::SolidTypes<double>::Transform camera_H_WorldAfter = worldBefore_H_cam.inversed()*worldBefore_R_worldAfter;

    positionCam = camera_H_WorldAfter.inversed().getOrigin();
    orientationCam = camera_H_WorldAfter.inversed().getOrientation();

    p_lookAt.setValue(getLookAtFromOrientation(positionCam, p_distance.getValue(), orientationCam));
    currentLookAt = p_lookAt.getValue();

    p_orientation.setValue(orientationCam);
    p_position.endEdit();
}

void Camera::moveCamera(int x, int y)
{
    float x1, x2, y1, y2;
    Quat newQuat;
    const unsigned int widthViewport = p_widthViewport.getValue();
    const unsigned int heightViewport = p_heightViewport.getValue();

    if (isMoving)
    {
        if (currentMode == TRACKBALL_MODE)
        {
            x1 = (2.0f * widthViewport / 2.0f - widthViewport) / widthViewport;
            y1 = (heightViewport- 2.0f *heightViewport / 2.0f) /heightViewport;
            x2 = (2.0f * (x + (-lastMousePosX + widthViewport / 2.0f)) - widthViewport) / widthViewport;
            y2 = (heightViewport- 2.0f * (y + (-lastMousePosY +heightViewport / 2.0f))) /heightViewport;

            currentTrackball.ComputeQuaternion(x1, y1, x2, y2);
            //fetch rotation
            newQuat = currentTrackball.GetQuaternion();

            Vec3 pivot = Vec3(0.0, 0.0, 0.0);
            //pivot = p_lookAt.getValue();
            pivot = sceneCenter;
            //pivot = (Vec3(x,y, p_distance.getValue()));
            //std::cout << "Pivot : " <<  pivot << std::endl;
            //rotateCameraAroundPoint(newQuat, pivot);
            rotateWorldAroundPoint(newQuat, pivot);
        }
        else if (currentMode == ZOOM_MODE)
        {
            Vec3 trans(0.0, 0.0, -p_zoomSpeed.getValue() * (y - lastMousePosY) / heightViewport);
            trans = cameraToWorldTransform(trans);
            translate(trans);
            translateLookAt(trans);
        }
        else if (currentMode == PAN_MODE)
        {
            Vec3 trans(lastMousePosX - x,  y-lastMousePosY, 0.0);
            trans = cameraToWorldTransform(trans)*p_panSpeed.getValue();
            translate(trans);
            translateLookAt(trans);
        }
        //must call update afterwards

        lastMousePosX = x;
        lastMousePosY = y;
    }
    else if (currentMode == WHEEL_ZOOM_MODE)
    {
        Vec3 trans(0.0, 0.0, -p_zoomSpeed.getValue() * (y*0.5) / heightViewport);
        trans = cameraToWorldTransform(trans);
        translate((trans));
        translateLookAt(trans);
        currentMode = NONE_MODE;
    }

    computeZ();
}

void Camera::computeZ()
{
    //if (!p_zNear.isSet() || !p_zFar.isSet())
    {
        double zNear = 1e10;
        double zFar = -1e10;
        double zNearTemp = zNear;
        double zFarTemp = zFar;

        const Vec3& currentPosition = getPosition();
        Quat currentOrientation = this->getOrientation();

        //double &zoomSpeed = *p_zoomSpeed.beginEdit();
        //double &panSpeed = *p_panSpeed.beginEdit();
        const Vec3 & minBBox = p_minBBox.getValue();
        const Vec3 & maxBBox = p_maxBBox.getValue();

        currentOrientation.normalize();
        helper::gl::Transformation transform;

        currentOrientation.buildRotationMatrix(transform.rotation);
        for (unsigned int i=0 ; i< 3 ; i++)
            transform.translation[i] = -currentPosition[i];

        for (int corner=0; corner<8; ++corner)
        {
            Vec3 p((corner&1)?minBBox[0]:maxBBox[0],
                    (corner&2)?minBBox[1]:maxBBox[1],
                    (corner&4)?minBBox[2]:maxBBox[2]);
            //TODO: invert transform...
            p = transform * p;
            double z = -p[2];
            if (z < zNearTemp) zNearTemp = z;
            if (z > zFarTemp)  zFarTemp = z;
        }

        //get the same zFar and zNear calculations as QGLViewer
        sceneCenter = (minBBox + maxBBox)*0.5;

        double distanceCamToCenter = (currentPosition - sceneCenter).norm();
        double zClippingCoeff = 3.5;
        double zNearCoeff = 0.005;
        double sceneRadius = (fabs(zFarTemp-zNearTemp))*0.5;

        zFar = distanceCamToCenter + zClippingCoeff*sceneRadius ;
        zNear = distanceCamToCenter- zClippingCoeff*sceneRadius;

        double zMin = zNearCoeff * zClippingCoeff * sceneRadius;
        if (zNear < zMin)
            zNear = zMin;

        zNear = 0.1;
        zFar = 1000.0;
        //update Speeds
        //zoomSpeed = sceneRadius;
        //panSpeed = sceneRadius;

        p_zNear.setValue(zNear);
        p_zFar.setValue(zFar);
        //p_zoomSpeed.endEdit();
        //p_panSpeed.endEdit();
    }
}

void Camera::manageEvent(core::objectmodel::Event* e)
{
    core::objectmodel::MouseEvent* me;
    core::objectmodel::KeypressedEvent* kpe;
    core::objectmodel::KeyreleasedEvent* kre;

    //Dispatch event
    if ((me = dynamic_cast<core::objectmodel::MouseEvent* > (e)))
        processMouseEvent(me);
    else if ((kpe = dynamic_cast<core::objectmodel::KeypressedEvent* > (e)))
        processKeyPressedEvent(kpe);
    else if ((kre = dynamic_cast<core::objectmodel::KeyreleasedEvent* > (e)))
        processKeyReleasedEvent(kre);
}

void Camera::processMouseEvent(core::objectmodel::MouseEvent* me)
{
    int posX = me->getPosX();
    int posY = me->getPosY();
    int wheelDelta = me->getWheelDelta();
    //Vec3 &camPosition = *p_position.beginEdit();

    //Mouse Press
    if(me->getState() == core::objectmodel::MouseEvent::LeftPressed)
    {
        isMoving = true;
        currentMode = TRACKBALL_MODE;
        lastMousePosX = posX;
        lastMousePosY = posY;
    }
    else if(me->getState() == core::objectmodel::MouseEvent::RightPressed)
    {
        isMoving = true;
        currentMode = PAN_MODE;
        lastMousePosX = posX;
        lastMousePosY = posY;
    }
    else if(me->getState() == core::objectmodel::MouseEvent::MiddlePressed)
    {
        isMoving = true;
        currentMode = ZOOM_MODE;
        lastMousePosX = posX;
        lastMousePosY = posY;
    }
    else
        //Mouse Move
        if(me->getState() == core::objectmodel::MouseEvent::Move)
        {
            //isMoving = true;
        }
        else
            //Mouse Release
            if(me->getState() == core::objectmodel::MouseEvent::LeftReleased)
            {
                isMoving = false;
                currentMode = NONE_MODE;
            }
            else if(me->getState() == core::objectmodel::MouseEvent::RightReleased)
            {
                isMoving = false;
                currentMode = NONE_MODE;
            }
            else if(me->getState() == core::objectmodel::MouseEvent::MiddleReleased)
            {
                isMoving = false;
                currentMode = NONE_MODE;
            }
    //Mouse Wheel
    if(me->getState() == core::objectmodel::MouseEvent::Wheel)
    {
        isMoving = false;
        currentMode = WHEEL_ZOOM_MODE;
        posX = 0;
        posY = wheelDelta;
    }

    moveCamera(posX, posY);

    p_position.endEdit();

}

void Camera::processKeyPressedEvent(core::objectmodel::KeypressedEvent*  /* kpe */)
{
    /*char keyPressed = kpe->getKey();

    switch(keyPressed)
    {
    	case 'a':
    	case 'A':
    	{
    		glPushMatrix();
    		//glLoadIdentity();
    		//helper::gl::Axis(p_position.getValue(), p_orientation.getValue(), 10.0);
    		glPopMatrix();
    		break;
    	}
    	default:
    	{
    		break;
    	}
    }
    */
}

void Camera::processKeyReleasedEvent(core::objectmodel::KeyreleasedEvent* /* kre */)
{

}

} // namespace visualmodel

} //namespace component

} //namespace sofa

