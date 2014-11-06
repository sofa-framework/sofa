/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_VISUALMODEL_BASECAMERA_H
#define SOFA_COMPONENT_VISUALMODEL_BASECAMERA_H


#include <sofa/core/objectmodel/BaseObject.h>
#include<sofa/SofaBase.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/Quater.h>
#include <sofa/helper/gl/Trackball.h>
#include <sofa/helper/gl/Transformation.h>

#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <sofa/core/objectmodel/MouseEvent.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/system/config.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

class SOFA_BASE_VISUAL_API BaseCamera : public core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(BaseCamera, core::objectmodel::BaseObject);

    typedef sofa::core::visual::VisualParams::CameraType CameraType;
    typedef defaulttype::Vector3 Vec3;
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



    Data<bool> p_activated;
	Data<bool> p_fixedLookAtPoint;

    BaseCamera();
    virtual ~BaseCamera();

    virtual void init();
    virtual void reinit();
    virtual void bwdInit();

    void activate();
    void desactivate();
    bool isActivated();

    bool exportParametersInFile(const std::string& viewFilename);
    bool importParametersFromFile(const std::string& viewFilename);

    void translate(const Vec3& t);
    void translateLookAt(const Vec3& t);
    void rotate(const Quat& r);
    void moveCamera(const Vec3 &p, const Quat &q);

    void rotateCameraAroundPoint( Quat& rotation, const Vec3& point);
    virtual void rotateWorldAroundPoint( Quat& rotation, const Vec3& point, Quat orientationCam);

    Vec3 cameraToWorldCoordinates(const Vec3& p);
    Vec3 worldToCameraCoordinates(const Vec3& p);
    Vec3 cameraToWorldTransform(const Vec3& v);
    Vec3 worldToCameraTransform(const Vec3& v);
    Vec3 screenToWorldCoordinates(int x, int y);


    void fitSphere(const Vec3& center, SReal radius);
    void fitBoundingBox(const Vec3& min,const Vec3& max);


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

    double getHorizontalFieldOfView()
    {
#ifndef SOFA_NO_OPENGL
        GLint viewport[4];
        glGetIntegerv( GL_VIEWPORT, viewport );
        float screenwidth = (float)viewport[2];
        float screenheight = (float)viewport[3];
        float aspectRatio = screenwidth / screenheight;
        float fov_radian = (float)getFieldOfView()* (float)(M_PI/180);
        float hor_fov_radian = 2.0f * atan ( tan(fov_radian/2.0f) * aspectRatio );
        return hor_fov_radian*(180/M_PI);
#else
	    return 0.0;
#endif /* SOFA_NO_OPENGL */
    }

    int getCameraType() const
    {
        return p_type.getValue();
    }

    void setCameraType(int type)
    {
        if (type == core::visual::VisualParams::ORTHOGRAPHIC_TYPE)
            p_type.setValue(core::visual::VisualParams::ORTHOGRAPHIC_TYPE);
        else
            p_type.setValue(core::visual::VisualParams::PERSPECTIVE_TYPE);
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

    void getOpenGLMatrix(GLdouble mat[16]);

    Quat getOrientationFromLookAt(const Vec3 &pos, const Vec3& lookat);
    Vec3 getLookAtFromOrientation(const Vec3 &pos, const double &distance,const Quat & orientation);
    Vec3 getPositionFromOrientation(const Vec3 &lookAt, const double &distance, const Quat& orientation);

    virtual void manageEvent(core::objectmodel::Event* e)=0;
    virtual void internalUpdate() {}

    void handleEvent(sofa::core::objectmodel::Event* event);
    void computeZ();
protected:
    Vec3 sceneCenter;
    SReal sceneRadius;


    //need to keep "internal" lookAt and distance for updating Data
    //better way to do that ?
    Vec3 currentLookAt;
    double currentDistance;
    double currentZNear, currentZFar;
};



} // namespace visualmodel

} // namespace component

} // namespace sofa

#endif //SOFA_COMPONENT_VISUALMODEL_BASECAMERA_H
