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

#ifndef SOFA_COMPONENT_VISUALMODEL_BASECAMERA_H
#define SOFA_COMPONENT_VISUALMODEL_BASECAMERA_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/component/component.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/Quater.h>
#include <sofa/helper/gl/Trackball.h>
#include <sofa/helper/gl/Transformation.h>

#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <sofa/core/objectmodel/MouseEvent.h>

class TiXmlElement;

namespace sofa
{

namespace component
{

namespace visualmodel
{

class SOFA_COMPONENT_VISUALMODEL_API BaseCamera : public core::objectmodel::BaseObject
{
public:
    enum  CameraType { PERSPECTIVE_TYPE =0, ORTHOGRAPHIC_TYPE =1};

    typedef defaulttype::Vec3d Vec3;
    typedef defaulttype::Quatd Quat;

    Data<Vec3> p_position;
    Data<Quat> p_orientation;
    Data<Vec3> p_lookAt;
    Data<double> p_distance;

    Data<double> p_fieldOfView;
    Data<double> p_zNear, p_zFar;
    Data<Vec3> p_minBBox, p_maxBBox;
    Data<unsigned int> p_widthViewport, p_heightViewport;
    Data<int> p_type;

    BaseCamera();
    virtual ~BaseCamera();

    virtual void init();
    virtual void reinit();

    bool exportParametersInFile(const std::string& viewFilename);
    bool importParametersFromFile(const std::string& viewFilename);

    void translate(const Vec3& t);
    void translateLookAt(const Vec3& t);
    void rotate(const Quat& r);
    void moveCamera(const Vec3 &p, const Quat &q);

    void rotateCameraAroundPoint( Quat& rotation, const Vec3& point);
    void rotateWorldAroundPoint( Quat& rotation, const Vec3& point);

    Vec3 cameraToWorldCoordinates(const Vec3& p);
    Vec3 worldToCameraCoordinates(const Vec3& p);
    Vec3 cameraToWorldTransform(const Vec3& v);
    Vec3 worldToCameraTransform(const Vec3& v);

    Vec3 getPosition()
    {
        return p_position.getValue();
    }

    Quat getOrientation()
    {
        if(currentLookAt !=  p_lookAt.getValue())
        {
            Quat newOrientation = getOrientationFromLookAt(p_position.getValue(), p_lookAt.getValue());
            p_orientation.setValue(newOrientation);

            currentLookAt = p_lookAt.getValue();
        }

        return p_orientation.getValue();
    }

    Vec3 getLookAt()
    {
        return p_lookAt.getValue();
    }

    double getDistance()
    {
        p_distance.setValue((p_lookAt.getValue() - p_position.getValue()).norm());
        return p_distance.getValue();
    }

    double getFieldOfView()
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

    double getZNear()
    {
        return currentZNear;
    }

    double getZFar()
    {
        return currentZFar;
    }

    void setView(const Vec3& position, const Quat &orientation);

    //Camera will look at the center of the scene's bounding box
    //at a good distance to view all the scene. The up vector will
    //be according to the gravity.
    void setDefaultView(const Vec3& gravity = Vec3(0, -9.81, 0));

    void getOpenGLMatrix(double mat[16]);

    Quat getOrientationFromLookAt(const Vec3 &pos, const Vec3& lookat);
    Vec3 getLookAtFromOrientation(const Vec3 &pos, const double &distance,const Quat & orientation);
    Vec3 getPositionFromOrientation(const Vec3 &lookAt, const double &distance, const Quat& orientation);

    virtual void manageEvent(core::objectmodel::Event* e)=0;
    virtual void internalUpdate() {}

    void handleEvent(sofa::core::objectmodel::Event* event);
protected:
    Vec3 sceneCenter;

    //need to keep "internal" lookAt and distance for updating Data
    //better way to do that ?
    Vec3 currentLookAt;
    double currentDistance;
    double currentZNear, currentZFar;

    void computeZ();

    void exportSingleParameter(TiXmlElement* root, core::objectmodel::BaseData& data, const std::string& comments = std::string());
    bool importSingleParameter(TiXmlElement* root, core::objectmodel::BaseData& data);
};



} // namespace visualmodel

} // namespace component

} // namespace sofa

#endif //SOFA_COMPONENT_VISUALMODEL_BASECAMERA_H
