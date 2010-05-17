/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
/*
 * FrameBufferObject.h
 *
 *  Created on: 6 janv. 2009
 *      Author: froy
 */

#ifndef SOFA_COMPONENT_VISUALMODEL_CAMERA_H
#define SOFA_COMPONENT_VISUALMODEL_CAMERA_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/component/component.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/Quater.h>
#include <sofa/helper/gl/Trackball.h>
#include <sofa/helper/gl/Transformation.h>

#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <sofa/core/objectmodel/MouseEvent.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

class SOFA_COMPONENT_VISUALMODEL_API Camera : public virtual core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(Camera, core::objectmodel::BaseObject);

    enum  { TRACKBALL_MODE, PAN_MODE, ZOOM_MODE, WHEEL_ZOOM_MODE, NONE_MODE };
    enum  CameraType { PERSPECTIVE_TYPE =0,	ORTHOGRAPHIC_TYPE =1};

    typedef defaulttype::Vec3d Vec3;
    typedef defaulttype::Quat Quat;

    Data<Vec3> p_position;
    Data<Quat> p_orientation;
    Data<Vec3> p_lookAt;
    Data<double> p_distance;

    Data<double> p_fieldOfView;
    Data<double> p_zNear, p_zFar;
    Data<Vec3> p_minBBox, p_maxBBox;
    Data<unsigned int> p_widthViewport, p_heightViewport;
    Data<int> p_type;
    Data<double> p_zoomSpeed;
    Data<double> p_panSpeed;

    Camera();
    virtual ~Camera();

    void init();

    Vec3 getPosition()
    {
        Vec3 newPos = computePosition(p_lookAt.getValue(), p_orientation.getValue());
        p_position.setValue(newPos);
        return newPos;
    }

    Quat getOrientation() const
    {
        return p_orientation.getValue();
    }

    Vec3 getLookAt() const
    {
        return p_lookAt.getValue();
    }

    double getDistance() const
    {
        return p_distance.getValue();
    }

    double getFieldOfView() const
    {
        return p_fieldOfView.getValue();
    }

    int getCameraType() const
    {
        return p_type.getValue();
    }

    void setCameraType(int type)
    {
        if (type == ORTHOGRAPHIC_TYPE)
            p_type.setValue(ORTHOGRAPHIC_TYPE);
        else
            p_type.setValue(PERSPECTIVE_TYPE);
    }


    void setBoundingBox(const Vec3 &min, const Vec3 &max)
    {
        p_minBBox.setValue(min);
        p_maxBBox.setValue(max);
        computeZ();
    }

    void setViewport(unsigned int w, unsigned int h)
    {
        p_widthViewport.setValue(w);
        p_heightViewport.setValue(h);
    }

    void setView(const Vec3& lookAt, const Quat &orientation)
    {
        p_lookAt.setValue(lookAt);
        p_orientation.setValue(orientation);
        computeZ();
    }

    void moveCamera(const Vec3 &position, const Quat &orientation);

    double getZNear() { return p_zNear.getValue(); }
    double getZFar() { return p_zFar.getValue(); }

    void manageEvent(core::objectmodel::Event* e);

protected:
    helper::gl::Transformation currentTransformation;
    int currentMode;
    bool isMoving;
    int lastMousePosX, lastMousePosY;
    helper::gl::Trackball currentTrackball;

    void processMouseEvent(core::objectmodel::MouseEvent* me);
    void processKeyPressedEvent(core::objectmodel::KeypressedEvent* kpe);
    void processKeyReleasedEvent(core::objectmodel::KeyreleasedEvent* kre);

    void moveCamera(int x, int y);

    void computeZ();

    Vec3 computeLookAt(const Vec3 &pos, const Quat& orientation);
    Vec3 computePosition(const Vec3 &lookAt, const Quat& orientation);
    void updateSpeed();
};



} // namespace visualmodel

} // namespace component

} // namespace sofa

#endif //SOFA_COMPONENT_VISUALMODEL_CAMERA_H
