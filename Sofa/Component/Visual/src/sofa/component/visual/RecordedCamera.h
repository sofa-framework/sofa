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

#include <sofa/component/visual/BaseCamera.h>
#include <sofa/helper/visual/Trackball.h>
#include <sofa/core/objectmodel/MouseEvent.h>

namespace sofa::component::visual
{

class SOFA_COMPONENT_VISUAL_API RecordedCamera : public BaseCamera
{
public:
    SOFA_CLASS(RecordedCamera, BaseCamera);

    SOFA_ATTRIBUTE_REPLACED__TYPEMEMBER(Vec3, sofa::type::Vec3);

    typedef BaseCamera::Quat Quat;
protected:
    RecordedCamera();
    ~RecordedCamera() override {}
public:
    void init() override;
    void reinit() override;
    void reset() override;
    void handleEvent(sofa::core::objectmodel::Event *) override;

    enum  { TRACKBALL_MODE, PAN_MODE, ZOOM_MODE, WHEEL_ZOOM_MODE, NONE_MODE };
    enum  { SCENE_CENTER_PIVOT = 0, WORLD_CENTER_PIVOT = 1};

    Data<double> d_zoomSpeed; ///< Zoom Speed
    Data<double> d_panSpeed; ///< Pan Speed
    Data<int> d_pivot; ///< Pivot (0 => Scene center, 1 => World Center

    void draw(const core::visual::VisualParams* vparams) override;

private:
    int currentMode;
    bool isMoving;
    int lastMousePosX, lastMousePosY;
    sofa::helper::visual::Trackball currentTrackball;

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

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_VISUAL()
    Data<double> p_zoomSpeed;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_VISUAL()
    Data<double> p_panSpeed;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_VISUAL()
    Data<int> p_pivot;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_VISUAL()
    Data<SReal> m_startTime;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_VISUAL()
    Data<SReal> m_endTime;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_VISUAL()
    Data<bool> m_rotationMode;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_VISUAL()
    Data<bool> m_translationMode;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_VISUAL()
    Data<bool> m_navigationMode;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_VISUAL()
    Data<SReal> m_rotationSpeed;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_VISUAL()
    Data<type::Vec3> m_rotationCenter;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_VISUAL()
    Data<type::Vec3> m_rotationStartPoint;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_VISUAL()
    Data<type::Vec3> m_rotationLookAt;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_VISUAL()
    Data<type::Vec3> m_rotationAxis;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_VISUAL()
    Data<type::Vec3> m_cameraUp;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_VISUAL()
    Data<bool> p_drawRotation;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_VISUAL()
    Data<bool> p_drawTranslation;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_VISUAL()
    Data<type::vector<type::Vec3>> m_translationPositions;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_VISUAL()    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_VISUAL()
    Data <sofa::type::vector<Quat> > m_translationOrientations;

    Data<SReal> d_startTime; ///< Time when the camera moves will start
    Data<SReal> d_endTime; ///< Time when the camera moves will end (or loop)

    Data <bool> d_rotationMode; ///< If true, rotation will be performed
    Data <bool> d_translationMode; ///< If true, translation will be performed
    Data <bool> d_navigationMode; ///< If true, navigation will be performed
    Data <SReal> d_rotationSpeed; ///< rotation Speed
    Data <type::Vec3> d_rotationCenter; ///< Rotation center coordinates
    Data <type::Vec3> d_rotationStartPoint; ///< Rotation start position coordinates
    Data <type::Vec3> d_rotationLookAt; ///< Position to be focused during rotation
    Data <type::Vec3> d_rotationAxis; ///< Rotation axis
    Data <type::Vec3> d_cameraUp; ///< Camera Up axis

    Data <bool> d_drawRotation; ///< If true, will draw the rotation path
    Data <bool> d_drawTranslation; ///< If true, will draw the translation path

    Data <sofa::type::vector<type::Vec3> > d_translationPositions; ///< Intermediate camera's positions
    Data <sofa::type::vector<Quat> > d_translationOrientations; ///< Intermediate camera's orientations

protected:
    double m_nextStep;
    double m_angleStep;
    bool firstIterationforRotation;
    bool firstIterationforTranslation;
    bool firstIterationforNavigation;

    sofa::type::vector<type::Vec3> m_rotationPoints;
};

} // namespace sofa::component::visual
