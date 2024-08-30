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
#include <sofa/component/visual/RecordedCamera.h>

#include <sofa/core/visual/VisualParams.h>
#include <sofa/type/RGBAColor.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>

namespace sofa::component::visual
{

void registerRecordedCamera(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("A camera that is moving along a predetermined path.")
        .add< RecordedCamera >());
}

RecordedCamera::RecordedCamera()
    : d_zoomSpeed(initData(&d_zoomSpeed, (double) 250.0 , "zoomSpeed", "Zoom Speed"))
    , d_panSpeed(initData(&d_panSpeed, (double) 0.1 , "panSpeed", "Pan Speed"))
    , d_pivot(initData(&d_pivot, 0 , "pivot", "Pivot (0 => Scene center, 1 => World Center"))
    , currentMode(RecordedCamera::NONE_MODE)
    , isMoving(false)
    , d_startTime(initData(&d_startTime, (SReal) 0.0 , "startTime", "Time when the camera moves will start"))
    , d_endTime(initData(&d_endTime, (SReal)200 , "endTime", "Time when the camera moves will end (or loop)"))
    , d_rotationMode(initData(&d_rotationMode, (bool)false , "rotationMode", "If true, rotation will be performed"))
    , d_translationMode(initData(&d_translationMode, (bool)false , "translationMode", "If true, translation will be performed"))
    , d_navigationMode(initData(&d_navigationMode, (bool)false , "navigationMode", "If true, navigation will be performed"))
    , d_rotationSpeed(initData(&d_rotationSpeed, (SReal)0.1 , "rotationSpeed", "rotation Speed"))
    , d_rotationCenter(initData(&d_rotationCenter, "rotationCenter", "Rotation center coordinates"))
    , d_rotationStartPoint(initData(&d_rotationStartPoint, "rotationStartPoint", "Rotation start position coordinates"))
    , d_rotationLookAt(initData(&d_rotationLookAt, "rotationLookAt", "Position to be focused during rotation"))
    , d_rotationAxis(initData(&d_rotationAxis, type::Vec3(0, 1, 0), "rotationAxis", "Rotation axis"))
    , d_cameraUp(initData(&d_cameraUp, type::Vec3(0, 0, 0), "cameraUp", "Camera Up axis"))
    , d_drawRotation(initData(&d_drawRotation, (bool)false , "drawRotation", "If true, will draw the rotation path"))
    , d_drawTranslation(initData(&d_drawTranslation, (bool)false , "drawTranslation", "If true, will draw the translation path"))
    , d_translationPositions(initData(&d_translationPositions, "cameraPositions", "Intermediate camera's positions"))
    , d_translationOrientations(initData(&d_translationOrientations, "cameraOrientations", "Intermediate camera's orientations"))
    , m_nextStep(0.0)
    , m_angleStep(0.0)
    , firstIterationforRotation(true)
    , firstIterationforTranslation(true)
    , firstIterationforNavigation(true)
{
    this->f_listening.setValue(true);

    p_zoomSpeed.setOriginalData(&d_zoomSpeed);
    p_panSpeed.setOriginalData(&d_panSpeed);
    p_pivot.setOriginalData(&d_pivot);
    m_startTime.setOriginalData(&d_startTime);
    m_endTime.setOriginalData(&d_endTime);
    m_rotationMode.setOriginalData(&d_rotationMode);
    m_translationMode.setOriginalData(&d_translationMode);
    m_navigationMode.setOriginalData(&d_navigationMode);
    m_rotationSpeed.setOriginalData(&d_rotationSpeed);
    m_rotationCenter.setOriginalData(&d_rotationCenter);
    m_rotationStartPoint.setOriginalData(&d_rotationStartPoint);
    m_rotationLookAt.setOriginalData(&d_rotationLookAt);
    m_rotationAxis.setOriginalData(&d_rotationAxis);
    m_cameraUp.setOriginalData(&d_cameraUp);
    p_drawRotation.setOriginalData(&d_drawRotation);
    p_drawTranslation.setOriginalData(&d_drawTranslation);
    m_translationPositions.setOriginalData(&d_translationPositions);
    m_translationOrientations.setOriginalData(&d_translationOrientations);

}

void RecordedCamera::init()
{
    BaseCamera::init();

    if (!d_rotationCenter.isSet())
        d_rotationCenter = type::Vec3(0.0, 10.0, 0.0);

    if (!d_rotationStartPoint.isSet())
        d_rotationStartPoint = type::Vec3(0.0, 10.0, 50.0);

    m_nextStep = d_startTime.getValue();

    if (d_drawRotation.getValue())
        this->drawRotation();

}


void RecordedCamera::reinit()
{
    BaseCamera::reinit();

    if (d_drawRotation.getValue())
        this->drawRotation();
}

void RecordedCamera::reset()
{
    BaseCamera::reset();
    m_nextStep = d_startTime.getValue();
    if(d_rotationMode.getValue())
        this->configureRotation();

    if(d_translationMode.getValue())
        this->configureTranslation();

    if(d_navigationMode.getValue())
        this->configureNavigation();
}

void RecordedCamera::moveCamera_navigation()
{
    double simuTime = this->getContext()->getTime();
    SReal totalTime = this->d_endTime.getValue();
    simuTime -= d_startTime.getValue();
    totalTime -= d_startTime.getValue();

    if (totalTime == 0.0)
        totalTime = 200.0;

    if(d_translationPositions.getValue().size() > 1 && d_translationOrientations.getValue().size() == d_translationPositions.getValue().size())
    {
        Quat firstQuater, nextQuater, interpolateQuater;

        const unsigned int nbrPoints = (unsigned int)d_translationPositions.getValue().size();
        // Time for each segment
        const double timeBySegment = totalTime/(nbrPoints - 1);
        // the animation is the same modulo totalTime
        const double simuTimeModTotalTime = fmod((SReal) simuTime,(SReal) totalTime);
        const unsigned int currentIndexPoint = (unsigned int)floor(((SReal)simuTimeModTotalTime/(SReal)timeBySegment));
        const double ratio =  fmod((SReal)simuTimeModTotalTime,(SReal)timeBySegment)/(SReal)timeBySegment;

        if(currentIndexPoint < nbrPoints - 1)
        {
            const type::Vec3 _pos = d_translationPositions.getValue()[currentIndexPoint];
            const type::Vec3 cameraFocal = d_translationPositions.getValue()[currentIndexPoint + 1] - _pos;

            // Set camera's position: linear interpolation
            d_position.setValue(d_translationPositions.getValue()[currentIndexPoint] + cameraFocal * ratio);

            // Set camera's orientation: slerp quaternion interpolation
            firstQuater = d_translationOrientations.getValue()[currentIndexPoint];
            nextQuater =  d_translationOrientations.getValue()[currentIndexPoint + 1];
            interpolateQuater.slerp(firstQuater,nextQuater,ratio);
            this->d_orientation.setValue(interpolateQuater);

            d_lookAt.setValue(getLookAtFromOrientation(_pos, d_distance.getValue(), d_orientation.getValue()));

        }

        else if (currentIndexPoint == nbrPoints - 1 )
        {
            d_position.setValue(d_translationPositions.getValue()[currentIndexPoint]);
            d_orientation.setValue(d_translationOrientations.getValue()[currentIndexPoint]);
        }
    }
}


void RecordedCamera::moveCamera_rotation()
{
    // Compute angle from Dt
    double simuTime = this->getContext()->getTime();
    //double simuDT = this->getContext()->getDt();
    SReal totalTime = this->d_endTime.getValue();
    simuTime -= d_startTime.getValue();
    totalTime -= d_startTime.getValue();

    if (totalTime == 0.0)
        totalTime = 200.0;

    const double ratio = (simuTime / totalTime);
    m_angleStep = 2*M_PI * ratio;

    // Compute cartesian coordinates from cylindrical ones
    type::Vec3 _pos = d_rotationCenter.getValue();
    const type::Quat<double> q(d_rotationAxis.getValue(), m_angleStep);
    _pos += q.rotate(d_rotationStartPoint.getValue() - d_rotationCenter.getValue());
    d_position.setValue(_pos);

    // dV to compute circle tangente
    type::Vec3 _poskk;
    if (d_cameraUp.isSet() && d_cameraUp.getValue().norm() > 0.000001)
        _poskk = -cross(_pos - d_lookAt.getValue(), d_cameraUp.getValue());
    else
        _poskk = -cross(_pos - d_rotationCenter.getValue(), d_rotationAxis.getValue());

    // Compute orientation
    type::Vec3 zAxis = -(d_lookAt.getValue() - _pos);
    type::Vec3 yAxis = zAxis.cross(_poskk);
    type::Vec3 xAxis = yAxis.cross(zAxis);
    xAxis.normalize();
    yAxis.normalize();
    zAxis.normalize();

    Quat orientation  = Quat::createQuaterFromFrame(xAxis, yAxis, zAxis);
    orientation.normalize();

    d_orientation.setValue(orientation);

    return;
}


void RecordedCamera::moveCamera_translation()
{
    double simuTime = this->getContext()->getTime();
    SReal totalTime = this->d_endTime.getValue();
    simuTime -= d_startTime.getValue();
    totalTime -= d_startTime.getValue();

    if (totalTime == 0.0)
        totalTime = 200.0;

    if(d_translationPositions.isSet() && d_translationPositions.getValue().size() > 0)
    {
        const unsigned int nbrPoints = (unsigned int)d_translationPositions.getValue().size();
        const double timeBySegment = totalTime/(nbrPoints - 1);
        const double simuTimeModTotalTime = fmod((SReal)simuTime,(SReal)totalTime);
        const unsigned int currentIndexPoint = (unsigned int)floor((simuTimeModTotalTime/timeBySegment));
        const double ratio = fmod(simuTimeModTotalTime,timeBySegment)/timeBySegment;

        // if the view up vector was not initialized
        if (d_cameraUp.getValue().norm() < 1e-6)
        this->initializeViewUp();

        if(currentIndexPoint < nbrPoints - 1)
        {
            const type::Vec3 _pos = d_translationPositions.getValue()[currentIndexPoint];
            d_lookAt.setValue(d_translationPositions.getValue()[currentIndexPoint + 1]);
            const type::Vec3 cameraFocal = d_lookAt.getValue() - _pos;

            // Set camera's position: linear interpolation
            d_position.setValue(d_translationPositions.getValue()[currentIndexPoint] + cameraFocal * ratio);

            // Set camera's orientation
            type::Vec3 zAxis = - (d_lookAt.getValue() - _pos);
            type::Vec3 xAxis = d_cameraUp.getValue().cross(zAxis);
            type::Vec3 yAxis = zAxis.cross(xAxis);
            xAxis.normalize();
            yAxis.normalize();
            zAxis.normalize();

            d_cameraUp.setValue(yAxis);
            Quat orientation  = Quat::createQuaterFromFrame(xAxis, yAxis, zAxis);
            orientation.normalize();
            d_orientation.setValue(orientation);
        }

        else if (currentIndexPoint == nbrPoints - 1 )
        {
            d_position.setValue(d_translationPositions.getValue()[currentIndexPoint]);
            d_lookAt.setValue(d_translationPositions.getValue()[currentIndexPoint]);
        }
    }

    return;
}


void RecordedCamera::handleEvent(sofa::core::objectmodel::Event *event)
{
    if (simulation::AnimateBeginEvent::checkEventType(event))
    {
        const double simuTime = this->getContext()->getTime();
        const double simuDT = this->getContext()->getDt();

        if (simuTime < m_nextStep)
            return;

        m_nextStep += simuDT;

        // init when start animation
       if(firstIterationforRotation & d_rotationMode.getValue())
            this->configureRotation();

        if(d_rotationMode.getValue())
            this->moveCamera_rotation();

        if (firstIterationforTranslation & d_translationMode.getValue())
            this->configureTranslation();

        if(d_translationMode.getValue())
            this->moveCamera_translation();

        if(firstIterationforNavigation & d_navigationMode.getValue())
            this->configureNavigation();

        if(d_navigationMode.getValue())
            this->moveCamera_navigation();
    }
    else if (sofa::core::objectmodel::KeypressedEvent::checkEventType(event))
    {
        const sofa::core::objectmodel::KeypressedEvent* ke = static_cast<sofa::core::objectmodel::KeypressedEvent*>(event);
        msg_info() <<" handleEvent gets character '" << ke->getKey() <<"'. ";
    }

}


void RecordedCamera::configureRotation()
{
    // HACK: need to init again, as component init seems to be overwritten by viewer settings
    const type::Vec3 _pos = d_rotationStartPoint.getValue();
    d_position.setValue(_pos);
    d_lookAt.setValue(d_rotationLookAt.getValue());
    d_distance.setValue((d_lookAt.getValue() - d_position.getValue()).norm());

    // dV to compute circle tangente
    type::Vec3 _poskk;
    if (d_cameraUp.isSet() && d_cameraUp.getValue().norm() > 0.000001)
        _poskk = -cross(_pos - d_lookAt.getValue(), d_cameraUp.getValue());
    else
        _poskk = -cross(_pos - d_rotationCenter.getValue(), d_rotationAxis.getValue());

    // Compute orientation
    type::Vec3 zAxis = -(d_lookAt.getValue() - _pos);
    type::Vec3 yAxis = zAxis.cross(_poskk);
    type::Vec3 xAxis = yAxis.cross(zAxis);
    xAxis.normalize();
    yAxis.normalize();
    zAxis.normalize();

    Quat orientation  = Quat::createQuaterFromFrame(xAxis, yAxis, zAxis);
    orientation.normalize();

    d_orientation.setValue(orientation);
    firstIterationforRotation = false;

    return;
}

void RecordedCamera::configureTranslation()
{
    if(d_translationPositions.isSet() && d_translationPositions.getValue().size() > 1)
    {
        // Set camera's position
        d_position.setValue(d_translationPositions.getValue()[0]);
        d_lookAt.setValue(d_translationPositions.getValue()[1]);

        // Set camera's orientation
        this->initializeViewUp();
        type::Vec3 zAxis = - d_translationPositions.getValue()[1] + d_translationPositions.getValue()[0];
        type::Vec3 yAxis = d_cameraUp.getValue();
        type::Vec3 xAxis = yAxis.cross(zAxis);
        xAxis.normalize();
        yAxis.normalize();
        zAxis.normalize();
        Quat orientation  = Quat::createQuaterFromFrame(xAxis, yAxis, zAxis);
        orientation.normalize();
        d_orientation.setValue(orientation);

        firstIterationforTranslation = false;
    }
    return;
}

void RecordedCamera:: configureNavigation()
{
    if(d_translationPositions.getValue().size() > 1 && d_translationOrientations.getValue().size() == d_translationPositions.getValue().size())
    {
        // Set camera's position
        d_position.setValue(d_translationPositions.getValue()[0]);

        // Set camera's orientation
        d_orientation.setValue(d_translationOrientations.getValue()[0]);

        firstIterationforNavigation = false;
    }
    return;
}

void RecordedCamera::initializeViewUp()
{
    if(d_translationPositions.isSet() && d_translationPositions.getValue().size() > 1)
    {
        type::Vec3 zAxis = d_translationPositions.getValue()[1] - d_translationPositions.getValue()[0];
        zAxis.normalize();
        const type::Vec3 xRef(1,0,0);
        // Initialize the view-up vector with the reference vector the "most perpendicular" to zAxis.
         d_cameraUp.setValue(xRef);
        double normCrossProduct = cross(zAxis,xRef).norm();
        for(int i = 1; i<3; ++ i)
        {
            type::Vec3 vecRef(0,0,0);
            vecRef[i] = 1;
            if(cross(zAxis,vecRef).norm() >= normCrossProduct )
            {
                normCrossProduct = cross(zAxis,vecRef).norm();
                d_cameraUp.setValue(vecRef);
            }
        }
    }
}

void RecordedCamera::manageEvent(core::objectmodel::Event* e)
{
    if(d_activated.getValue())
    {
        //Dispatch event
        if (sofa::core::objectmodel::MouseEvent::checkEventType(e))
        {
            sofa::core::objectmodel::MouseEvent* me = static_cast<sofa::core::objectmodel::MouseEvent*>(e);
            processMouseEvent(me);
        }
    }
    else
    {
        isMoving = false;
        currentMode = NONE_MODE;
    }
}


void RecordedCamera::processMouseEvent(core::objectmodel::MouseEvent* me)
{
    int posX = me->getPosX();
    int posY = me->getPosY();
    const int wheelDelta = me->getWheelDelta();

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

    moveCamera_mouse(posX, posY);

    d_position.endEdit();

}


void RecordedCamera::moveCamera_mouse(int x, int y)
{
    Quat newQuat;
    const unsigned int widthViewport = d_widthViewport.getValue();
    const unsigned int heightViewport = d_heightViewport.getValue();

    if (isMoving)
    {
        if (currentMode == TRACKBALL_MODE)
        {
            const float x1 = (2.0f * widthViewport / 2.0f - widthViewport) / widthViewport;
            const float y1 = (heightViewport- 2.0f *heightViewport / 2.0f) /heightViewport;
            const float x2 = (2.0f * (x + (-lastMousePosX + widthViewport / 2.0f)) - widthViewport) / widthViewport;
            const float y2 = (heightViewport- 2.0f * (y + (-lastMousePosY +heightViewport / 2.0f))) /heightViewport;
            currentTrackball.ComputeQuaternion(x1, y1, x2, y2);

            //fetch rotation
            newQuat = currentTrackball.GetQuaternion();
            type::Vec3 pivot;
            switch (d_pivot.getValue())
            {
            case WORLD_CENTER_PIVOT:
                pivot = type::Vec3(0.0, 0.0, 0.0);
                break;
            case SCENE_CENTER_PIVOT :
            default:
                pivot = sceneCenter;
                break;
            }

            BaseCamera::rotateWorldAroundPoint(newQuat, pivot, this->getOrientation());
        }
        else if (currentMode == ZOOM_MODE)
        {
            type::Vec3 trans(0.0, 0.0, -d_zoomSpeed.getValue() * (y - lastMousePosY) / heightViewport);
            trans = cameraToWorldTransform(trans);
            translate(trans);
            translateLookAt(trans);
        }
        else if (currentMode == PAN_MODE)
        {
            type::Vec3 trans(lastMousePosX - x,  y-lastMousePosY, 0.0);
            trans = cameraToWorldTransform(trans) * d_panSpeed.getValue();
            translate(trans);
            translateLookAt(trans);
        }
        //must call update afterwards

        lastMousePosX = x;
        lastMousePosY = y;
    }
    else if (currentMode == WHEEL_ZOOM_MODE)
    {
        type::Vec3 trans(0.0, 0.0, -d_zoomSpeed.getValue() * (y * 0.5) / heightViewport);
        trans = cameraToWorldTransform(trans);
        translate((trans));
        translateLookAt(trans);
        currentMode = NONE_MODE;
    }

    computeZ();
}


void RecordedCamera::drawRotation()
{
    type::Vec3 _pos = d_rotationStartPoint.getValue();

    m_rotationPoints.resize(100);
    const double _angleStep = 2*M_PI/100;
    for (unsigned int i = 0; i<100; ++i)
    {
        // Compute cartesian coordinates from cylindrical ones
        _pos = d_rotationCenter.getValue();
        type::Quat<double> q(d_rotationAxis.getValue(), _angleStep * i);
        _pos += q.rotate(d_rotationStartPoint.getValue() - d_rotationCenter.getValue());
        m_rotationPoints[i] = _pos;
    }

    return;
}

void RecordedCamera::draw(const core::visual::VisualParams* vparams)
{
    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    // Draw rotation path
    if(d_drawRotation.getValue())
    {
        if (m_rotationPoints.empty())
            return;

        vparams->drawTool()->disableLighting();
        static constexpr sofa::type::RGBAColor color(0.f,1.f,0.5f,1.f);
        std::vector<sofa::type::Vec3> vertices;

        // Camera positions
        for (unsigned int i=0; i<m_rotationPoints.size()-1; ++i)
        {
            vertices.emplace_back(m_rotationPoints[i  ][0], m_rotationPoints[i  ][1], m_rotationPoints[i  ][2]);
            vertices.emplace_back(m_rotationPoints[i+1][0], m_rotationPoints[i+1][1], m_rotationPoints[i+1][2]);
        }
        vertices.emplace_back(m_rotationPoints.back()[0], m_rotationPoints.back()[1], m_rotationPoints.back()[2]);
        vertices.emplace_back(m_rotationPoints[0    ][0], m_rotationPoints[0    ][1], m_rotationPoints[0    ][2]);

        vparams->drawTool()->drawLines(vertices,1,color);
        vertices.clear();

        const type::Vec3& _lookAt = d_rotationLookAt.getValue();
        static constexpr unsigned int dx = 4;
        const std::size_t ratio = m_rotationPoints.size()/dx;

        for (unsigned int i=0; i<dx; ++i)
        {
            vertices.emplace_back(m_rotationPoints[i*ratio][0], m_rotationPoints[i*ratio][1], m_rotationPoints[i*ratio][2]);
            vertices.emplace_back(_lookAt[0], _lookAt[1], _lookAt[2]);
        }
        vparams->drawTool()->drawLines(vertices,1,color);
    }

    // Draw translation path
    if(d_drawTranslation.getValue())
    {
        if (d_translationPositions.getValue().size() < 2)
            return;

        vparams->drawTool()->disableLighting();
        constexpr sofa::type::RGBAColor color(0,1,0.5,1);
        std::vector<sofa::type::Vec3> vertices;

        // Camera positions
        type::vector<type::Vec3> _positions = d_translationPositions.getValue();
        for (unsigned int i=0; i < _positions.size()-1; ++i)
        {
            vertices.push_back(sofa::type::Vec3((float)_positions[i  ][0], (float)_positions[i  ][1], (float)_positions[i  ][2]));
            vertices.push_back(sofa::type::Vec3((float)_positions[i+1][0], (float)_positions[i+1][1], (float)_positions[i+1][2]));
        }
        vparams->drawTool()->drawLines(vertices,1,color);
    }

}

} // namespace sofa::component::visual
