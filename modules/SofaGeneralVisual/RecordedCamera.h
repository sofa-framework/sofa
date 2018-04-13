/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef RECORDEDCAMERA_H
#define RECORDEDCAMERA_H
#include "config.h"

#include <SofaBaseVisual/BaseCamera.h>
#include <sofa/helper/gl/Trackball.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

class SOFA_GENERAL_VISUAL_API RecordedCamera : public BaseCamera
{
public:
    SOFA_CLASS(RecordedCamera, BaseCamera);

    typedef BaseCamera::Vec3 Vec3;
    typedef BaseCamera::Quat Quat;
protected:
    RecordedCamera();
    virtual ~RecordedCamera() {}
public:
    virtual void init() override;

    virtual void reinit() override;

    virtual void reset() override;

    virtual void handleEvent(sofa::core::objectmodel::Event *) override;

    //virtual void rotateWorldAroundPoint(Quat &rotation, const Vec3 &point);

    enum  { TRACKBALL_MODE, PAN_MODE, ZOOM_MODE, WHEEL_ZOOM_MODE, NONE_MODE };
    enum  { SCENE_CENTER_PIVOT = 0, WORLD_CENTER_PIVOT = 1};

    Data<double> p_zoomSpeed; ///< Zoom Speed
    Data<double> p_panSpeed; ///< Pan Speed
    Data<int> p_pivot; ///< Pivot (0 => Scene center, 1 => World Center

    void draw(const core::visual::VisualParams* vparams) override;

private:
    int currentMode;
    bool isMoving;
    int lastMousePosX, lastMousePosY;
    helper::gl::Trackball currentTrackball;

    void moveCamera_rotation();
    void moveCamera_translation();
    void moveCamera_navigation();

    // Kepp functions for mouse interaction (TODO: removed them and allow interactive and recorded camera in same scene)
    void moveCamera_mouse(int x, int y);
    void manageEvent(core::objectmodel::Event* e) override;
    void processMouseEvent(core::objectmodel::MouseEvent* me);

    void configureRotation();
    void configureTranslation();
    void configureNavigation();
    void initializeViewUp();
    void drawRotation();

public:
    Data<SReal> m_startTime; ///< Time when the camera moves will start
    Data<SReal> m_endTime; ///< Time when the camera moves will end (or loop)

    Data <bool> m_rotationMode; ///< If true, rotation will be performed
    Data <bool> m_translationMode; ///< If true, translation will be performed
    Data <bool> m_navigationMode; ///< If true, navigation will be performed
    Data <SReal> m_rotationSpeed; ///< rotation Speed
    Data <Vec3> m_rotationCenter; ///< Rotation center coordinates
    Data <Vec3> m_rotationStartPoint; ///< Rotation start position coordinates
    Data <Vec3> m_rotationLookAt; ///< Position to be focused during rotation
    Data <Vec3> m_rotationAxis; ///< Rotation axis
    Data <Vec3> m_cameraUp; ///< Camera Up axis

    Data <bool> p_drawRotation; ///< If true, will draw the rotation path
    Data <bool> p_drawTranslation; ///< If true, will draw the translation path

    Data <sofa::helper::vector<Vec3> > m_translationPositions; ///< Intermediate camera's positions
    Data <sofa::helper::vector<Quat> > m_translationOrientations; ///< Intermediate camera's orientations

protected:
    double m_nextStep;
    double m_angleStep;
    //double m_initAngle;
    //double m_radius;
    bool firstIterationforRotation;
    bool firstIterationforTranslation;
    bool firstIterationforNavigation;
  
    sofa::helper::vector <Vec3> m_rotationPoints;
};

} // namespace visualmodel

} // namespace component

} // namespace sofa

#endif // RecordedCamera_H
