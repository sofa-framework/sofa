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
    ,p_panSpeed(initData(&p_panSpeed, (double) 25.0 , "panSpeed", "Pan Speed"))
    ,currentMode(Camera::NONE_MODE)
    ,isMoving(false)
{

}

Camera::~Camera()
{

}

void Camera::init()
{
    currentTrackball.ComputeQuaternion(0.0, 0.0, 0.0, 0.0);

    if(p_position.isSet() && p_orientation.isSet())
    {
        Vec3 newLookAt = computeLookAt(p_position.getValue(), p_orientation.getValue());
        p_lookAt.setValue(newLookAt);
    }
    else if (p_lookAt.isSet() && p_orientation.isSet())
    {
        Vec3 newPos = computePosition(p_lookAt.getValue(), p_orientation.getValue());
        p_position.setValue(newPos);
    }
    else
    {
        serr << "Not enough parameters, putting default ..." << sendl;
    }
}

Camera::Vec3 Camera::computeLookAt(const Camera::Vec3 &pos, const Camera::Quat& orientation)
{
    Vec3 dirVec = orientation.rotate(Vec3(0,0,1));
    return dirVec - pos;
}

Camera::Vec3 Camera::computePosition(const Camera::Vec3 &lookAt, const Camera::Quat& orientation)
{
    return orientation.rotate(-lookAt);
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
        Quat currentOrientation = p_orientation.getValue();

        double &zoomSpeed = *p_zoomSpeed.beginEdit();
        double &panSpeed = *p_panSpeed.beginEdit();
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
        Vec3 center = (minBBox + maxBBox)*0.5;

        double distanceCamToCenter = (currentPosition - center).norm();
        double zClippingCoeff = 3.5;
        double zNearCoeff = 0.005;
        double sceneRadius = (fabs(zFarTemp-zNearTemp))*0.5;

        zFar = distanceCamToCenter + zClippingCoeff*sceneRadius ;
        zNear = distanceCamToCenter- zClippingCoeff*sceneRadius;

        float zMin = zNearCoeff * zClippingCoeff * sceneRadius;
        if (zNear < zMin)
            zNear = zMin;

        //update Speeds
        zoomSpeed = sceneRadius;
        panSpeed = sceneRadius;

        p_zNear.setValue(zNear);
        p_zFar.setValue(zFar);
        p_zoomSpeed.endEdit();
        p_panSpeed.endEdit();
    }
}

void Camera::moveCamera(int x, int y)
{
    float x1, x2, y1, y2;
    float xshift, yshift, zshift;
    Quat currentQuat = p_orientation.getValue();
    Quat newQuat;
    Vec3 &camPosition = *p_position.beginEdit();
    Vec3 &lookAtPosition = *p_lookAt.beginEdit();
    const unsigned int widthViewport = p_widthViewport.getValue();
    const unsigned int heightViewport = p_heightViewport.getValue();

    //std::cout << "moveCamera(int x, int y)" << x << " " << y << std::endl;

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

            currentQuat =  newQuat + currentQuat;
            camPosition =  currentQuat.rotate(-lookAtPosition);
            p_orientation.setValue(currentQuat);
        }
        else if (currentMode == ZOOM_MODE)
        {
            zshift = (2.0f * y - widthViewport) / widthViewport - (2.0f * lastMousePosY - widthViewport) / widthViewport;
            lookAtPosition[2] = lookAtPosition[2]+ p_zoomSpeed.getValue() * zshift;
        }
        else if (currentMode == PAN_MODE)
        {
            xshift = (2.0f * x - widthViewport) / widthViewport - (2.0f * lastMousePosX - widthViewport) / widthViewport;
            yshift = (2.0f * y - widthViewport) / widthViewport - (2.0f * lastMousePosY - widthViewport) / widthViewport;

            lookAtPosition[0] = lookAtPosition[0] + p_panSpeed.getValue() * xshift;
            lookAtPosition[1] = lookAtPosition[1] - p_panSpeed.getValue() * yshift;
        }
        //must call update afterwards

        lastMousePosX = x;
        lastMousePosY = y;
    }
    else if (currentMode == WHEEL_ZOOM_MODE)
    {
        zshift = (2.0f * y / widthViewport);
        lookAtPosition[2] = lookAtPosition[2]+ p_zoomSpeed.getValue() * zshift;
        currentMode = NONE_MODE;
    }

    computeZ();
    p_position.endEdit();
    p_lookAt.endEdit();
}

void Camera::moveCamera(const Vec3 &translation, const Quat &rotation)
{
    Quat &currentQuat = *p_orientation.beginEdit();
    Vec3 &lookAtPosition = *p_lookAt.beginEdit();
    Quat tempQuat = rotation;
    tempQuat.normalize();

    lookAtPosition += translation;
    currentQuat += tempQuat;

    p_lookAt.endEdit();
    p_orientation.endEdit();
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
    /*	char keyPressed = kpe->getKey();

    	switch(keyPressed)
    	{
    		case 't':
    		case 'T':
    		{
    			int type = (p_type.getValue() == PERSPECTIVE_TYPE) ? ORTHOGRAPHIC_TYPE : PERSPECTIVE_TYPE;
    			p_type.setValue(type);
    			break;
    		}
    		default:
    		{
    			break;
    		}
    	}*/

}

void Camera::processKeyReleasedEvent(core::objectmodel::KeyreleasedEvent* /* kre */)
{

}

} // namespace visualmodel

} //namespace component

} //namespace sofa
