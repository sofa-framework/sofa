/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once
#include <sofa/component/visual/config.h>

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/type/Vec.h>
#include <sofa/type/Ray.h>
#include <sofa/type/Quat.h>

#include <sofa/core/fwd.h>
#include <sofa/helper/OptionsGroup.h>

namespace sofa::component::visual
{

class SOFA_COMPONENT_VISUAL_API BaseCamera : public core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(BaseCamera, core::objectmodel::BaseObject);

    typedef type::Quat<SReal> Quat;

    enum Side {LEFT, RIGHT, MONO};

    enum StereoMode
    {
        STEREO_AUTO = 0,
        STEREO_INTERLACED = 1,
        STEREO_FRAME_PACKING = 2,
        STEREO_SIDE_BY_SIDE = 3,
        STEREO_TOP_BOTTOM = 4,
        STEREO_SIDE_BY_SIDE_HALF = 5,
        STEREO_TOP_BOTTOM_HALF = 6,
        STEREO_NONE = 7,
        NB_STEREO_MODES = 8
    };
    enum StereoStrategy
    {
        PARALLEL = 0,
        TOEDIN = 1

    };

    Data<type::Vec3> d_position; ///< Camera's position
    Data<Quat> d_orientation; ///< Camera's orientation
    Data<type::Vec3> d_lookAt; ///< Camera's look at
    Data<double> d_distance; ///< Distance between camera and look at

    Data<double> d_fieldOfView; ///< Camera's FOV
    Data<double> d_zNear; ///< Camera's zNear
    Data<double> d_zFar; ///< Camera's zFar
    Data<bool> d_computeZClip; ///< Compute Z clip planes (Near and Far) according to the bounding box
    Data<type::Vec3> d_minBBox; ///< minBBox
    Data<type::Vec3> d_maxBBox; ///< maxBBox
    Data<unsigned int> d_widthViewport; ///< widthViewport
    Data<unsigned int> d_heightViewport; ///< heightViewport
    Data<sofa::helper::OptionsGroup> d_type; ///< Camera Type (0 = Perspective, 1 = Orthographic)

    Data<bool> d_activated; ///< Camera activated ?
    Data<bool> d_fixedLookAtPoint; ///< keep the lookAt point always fixed

    Data<type::vector<SReal> > d_modelViewMatrix; ///< ModelView Matrix
    Data<type::vector<SReal> > d_projectionMatrix; ///< Projection Matrix

    BaseCamera();
    ~BaseCamera() override;

    void init() override;
    void reinit() override;
    void bwdInit() override;

    void activate();
    void desactivate();
    bool isActivated();

    bool exportParametersInFile(const std::string& viewFilename);
    bool importParametersFromFile(const std::string& viewFilename);

    void translate(const type::Vec3& t);
    void translateLookAt(const type::Vec3& t);
    void rotate(const Quat& r);
    void moveCamera(const type::Vec3 &p, const Quat &q);

    void rotateCameraAroundPoint( Quat& rotation, const type::Vec3& point);
    virtual void rotateWorldAroundPoint(Quat& rotation, const type::Vec3& point, Quat orientationCam);
    virtual void rotateWorldAroundPoint(Quat& rotation, const type::Vec3& point, Quat orientationCam, type::Vec3 positionCam);

    type::Vec3 screenToViewportPoint(const type::Vec3& p) const;
    type::Vec3 screenToWorldPoint(const type::Vec3& p);

    type::Vec3 viewportToScreenPoint(const type::Vec3& p) const;
    type::Vec3 viewportToWorldPoint(const type::Vec3& p);

    type::Vec3 worldToScreenPoint(const type::Vec3& p);
    type::Vec3 worldToViewportPoint(const type::Vec3& p);

    type::Ray viewportPointToRay(const type::Vec3&p);
    type::Ray screenPointToRay(const type::Vec3&p);

    type::Ray toRay() const;


    type::Vec3 cameraToWorldCoordinates(const type::Vec3& p);
    type::Vec3 worldToCameraCoordinates(const type::Vec3& p);
    type::Vec3 cameraToWorldTransform(const type::Vec3& v);
    type::Vec3 worldToCameraTransform(const type::Vec3& v);
    type::Vec3 screenToWorldCoordinates(int x, int y);
    type::Vec2 worldToScreenCoordinates(const type::Vec3& p);

    void fitSphere(const type::Vec3& center, SReal radius);
    void fitBoundingBox(const type::Vec3& min,const type::Vec3& max);


    type::Vec3 getPosition()
    {
        return d_position.getValue();
    }

    Quat getOrientation() ;
    type::Vec3 getLookAt()
    {
        return d_lookAt.getValue();
    }

    double getDistance()
    {
        d_distance.setValue((d_lookAt.getValue() - d_position.getValue()).norm());
        return d_distance.getValue();
    }

    double getFieldOfView()
    {
        return d_fieldOfView.getValue();
    }

    double getHorizontalFieldOfView() ;

    unsigned int getCameraType() const ;

    void setCameraType(unsigned int type) ;

    void setBoundingBox(const type::Vec3 &min, const type::Vec3 &max)
    {
        d_minBBox.setValue(min);
        d_maxBBox.setValue(max);

        sceneCenter = (min + max)*0.5;
        sceneRadius = 0.5*(max - min).norm();

        computeZ();
    }

    void setViewport(unsigned int w, unsigned int h)
    {
        d_widthViewport.setValue(w);
        d_heightViewport.setValue(h);
    }

    double getZNear()
    {
        return currentZNear;
    }

    double getZFar()
    {
        return currentZFar;
    }

    void setView(const type::Vec3& position, const Quat &orientation);

    //Camera will look at the center of the scene's bounding box
    //at a good distance to view all the scene. The up vector will
    //be according to the gravity.
    void setDefaultView(const type::Vec3& gravity = type::Vec3(0, -9.81, 0));

    virtual void getModelViewMatrix(double mat[16]);
    virtual void getProjectionMatrix(double mat[16]);
    void getOpenGLModelViewMatrix(double mat[16]);
    void getOpenGLProjectionMatrix(double mat[16]);

    Quat getOrientationFromLookAt(const type::Vec3 &pos, const type::Vec3& lookat);
    type::Vec3 getLookAtFromOrientation(const type::Vec3 &pos, const double &distance,const Quat & orientation);
    type::Vec3 getPositionFromOrientation(const type::Vec3 &lookAt, const double &distance, const Quat& orientation);

    virtual void manageEvent(core::objectmodel::Event* event) = 0 ;
    virtual void internalUpdate() {}

    void handleEvent(sofa::core::objectmodel::Event* event) override;
    void computeZ();

    virtual bool isStereo()
    {
        return false;
    }
    virtual void setCurrentSide(Side /*newSide*/)
    {
        return;
    }
    virtual Side getCurrentSide()
    {
        return MONO;
    }
    virtual void setStereoEnabled(bool /*newEnable*/)
    {
        return;
    }
    virtual bool getStereoEnabled()
    {
        return false;
    }
    virtual void setStereoMode(StereoMode /*newMode*/)
    {
        return;
    }
    virtual StereoMode getStereoMode()
    {
        return STEREO_AUTO;
    }
    virtual void setStereoStrategy(StereoStrategy /*newStrategy*/)
    {
        return;
    }
    virtual StereoStrategy getStereoStrategy()
    {
        return PARALLEL;
    }
    virtual void setStereoShift(double /*newShift*/)
    {
        return;
    }
    virtual double getStereoShift()
    {
        return 1.0;
    }


    void draw(const core::visual::VisualParams*) override ;
    void computeClippingPlane(const core::visual::VisualParams* vp, double& zNear, double& zFar);
    virtual void drawCamera(const core::visual::VisualParams*);
protected:
    void updateOutputData();

    type::Vec3 sceneCenter;
    SReal sceneRadius;

    bool b_setDefaultParameters;

    //need to keep "internal" lookAt and distance for updating Data
    //better way to do that ?
    type::Vec3 currentLookAt;
    double currentDistance;
    double currentZNear, currentZFar;
};

} // namespace sofa::component::visual
