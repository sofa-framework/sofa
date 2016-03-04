/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaBaseVisual/RecordedCamera.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/common/AnimateBeginEvent.h>
#include <sofa/simulation/common/AnimateEndEvent.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>

#include <math.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace sofa
{

namespace component
{

namespace visualmodel
{

SOFA_DECL_CLASS(RecordedCamera)

int RecordedCameraClass = core::RegisterObject("Camera moving along a predetermined path (currently only a rotation)")
        .add< RecordedCamera >()
        ;


RecordedCamera::RecordedCamera()
    : p_zoomSpeed(initData(&p_zoomSpeed, (double) 250.0 , "zoomSpeed", "Zoom Speed"))
    , p_panSpeed(initData(&p_panSpeed, (double) 0.1 , "panSpeed", "Pan Speed"))
    , p_pivot(initData(&p_pivot, 0 , "pivot", "Pivot (0 => Scene center, 1 => World Center"))
    , currentMode(RecordedCamera::NONE_MODE)
    , isMoving(false)
    , m_startTime(initData(&m_startTime, (SReal) 0.0 , "startTime", "Time when the camera moves will start"))
    , m_endTime(initData(&m_endTime, (SReal)200 , "endTime", "Time when the camera moves will end (or loop)"))
    , m_rotationMode(initData(&m_rotationMode, (bool)false , "rotationMode", "If true, rotation will be performed"))
    , m_translationMode(initData(&m_translationMode, (bool)false , "translationMode", "If true, translation will be performed"))
    , m_navigationMode(initData(&m_navigationMode, (bool)false , "navigationMode", "If true, navigation will be performed"))
    , m_rotationSpeed(initData(&m_rotationSpeed, (SReal)0.1 , "rotationSpeed", "rotation Speed"))
    , m_rotationCenter(initData(&m_rotationCenter, "rotationCenter", "Rotation center coordinates"))
    , m_rotationStartPoint(initData(&m_rotationStartPoint, "rotationStartPoint", "Rotation start position coordinates"))
    , m_rotationLookAt(initData(&m_rotationLookAt, "rotationLookAt", "Position to be focused during rotation"))
    , m_rotationAxis(initData(&m_rotationAxis, Vec3(0,1,0), "rotationAxis", "Rotation axis"))
    , m_cameraUp(initData(&m_cameraUp, Vec3(0,0,0), "cameraUp", "Camera Up axis"))
    , p_drawRotation(initData(&p_drawRotation, (bool)false , "drawRotation", "If true, will draw the rotation path"))
    , p_drawTranslation(initData(&p_drawTranslation, (bool)false , "drawTranslation", "If true, will draw the translation path"))
    , m_translationPositions(initData(&m_translationPositions, "cameraPositions", "Intermediate camera's positions"))
    , m_translationOrientations(initData(&m_translationOrientations, "cameraOrientations", "Intermediate camera's orientations"))
    , m_nextStep(0.0)
    , m_angleStep(0.0)
    //, m_initAngle(0.0)
    ,firstIterationforRotation(true)
    ,firstIterationforTranslation(true)
    ,firstIterationforNavigation(true)
{
    this->f_listening.setValue(true);
}

void RecordedCamera::init()
{
    BaseCamera::init();

    if (!m_rotationCenter.isSet())
        m_rotationCenter = Vec3(0.0, 10.0, 0.0);

    if (!m_rotationStartPoint.isSet())
        m_rotationStartPoint = Vec3(0.0, 10.0, 50.0);

    m_nextStep = m_startTime.getValue();

    if (p_drawRotation.getValue())
        this->drawRotation();

}


void RecordedCamera::reinit()
{
    BaseCamera::reinit();

    if (p_drawRotation.getValue())
        this->drawRotation();
}

void RecordedCamera::reset()
{
    BaseCamera::reset();
    m_nextStep = m_startTime.getValue();
    if(m_rotationMode.getValue())
        this->configureRotation();

    if(m_translationMode.getValue())
        this->configureTranslation();

    if(m_navigationMode.getValue())
        this->configureNavigation();
}

void RecordedCamera::moveCamera_navigation()
{
    double simuTime = this->getContext()->getTime();
    SReal totalTime = this->m_endTime.getValue();
    simuTime -= m_startTime.getValue();
    totalTime -= m_startTime.getValue();

    if (totalTime == 0.0)
        totalTime = 200.0;

    if(m_translationPositions.getValue().size() > 1 &&  m_translationOrientations.getValue().size() == m_translationPositions.getValue().size())
    {
        Quat firstQuater, nextQuater, interpolateQuater;

        unsigned int nbrPoints = (unsigned int)m_translationPositions.getValue().size();
        // Time for each segment
        double timeBySegment = totalTime/(nbrPoints - 1);
        // the animation is the same modulo totalTime
        double simuTimeModTotalTime = fmod((SReal) simuTime,(SReal) totalTime);
        unsigned int currentIndexPoint = (unsigned int)floor(((SReal)simuTimeModTotalTime/(SReal)timeBySegment));
        double ratio =  fmod((SReal)simuTimeModTotalTime,(SReal)timeBySegment)/(SReal)timeBySegment;

        if(currentIndexPoint < nbrPoints - 1)
        {
            Vec3 _pos = m_translationPositions.getValue()[currentIndexPoint];
            Vec3 cameraFocal = m_translationPositions.getValue()[currentIndexPoint + 1] - _pos;

            // Set camera's position: linear interpolation 
            p_position.setValue( m_translationPositions.getValue()[currentIndexPoint] + cameraFocal * ratio);

            // Set camera's orientation: slerp quaternion interpolation
            firstQuater = m_translationOrientations.getValue()[currentIndexPoint];
            nextQuater =  m_translationOrientations.getValue()[currentIndexPoint + 1];
            interpolateQuater.slerp(firstQuater,nextQuater,ratio);
            this->p_orientation.setValue(interpolateQuater);

            p_lookAt.setValue(getLookAtFromOrientation(_pos, p_distance.getValue(), p_orientation.getValue()));

        }

        else if (currentIndexPoint == nbrPoints - 1 )
        {
            p_position.setValue(m_translationPositions.getValue()[currentIndexPoint]);
            p_orientation.setValue(m_translationOrientations.getValue()[currentIndexPoint]);
        }
    }
}


void RecordedCamera::moveCamera_rotation()
{
    // Compute angle from Dt
    double simuTime = this->getContext()->getTime();
    //double simuDT = this->getContext()->getDt();
    SReal totalTime = this->m_endTime.getValue();
    simuTime -= m_startTime.getValue();
    totalTime -= m_startTime.getValue();

    if (totalTime == 0.0)
        totalTime = 200.0;

    double ratio = (simuTime / totalTime);
    m_angleStep = 2*M_PI * ratio;

    // Compute cartesian coordinates from cylindrical ones
    Vec3 _pos = m_rotationCenter.getValue();
    helper::Quater<double> q(m_rotationAxis.getValue(), m_angleStep);
    _pos += q.rotate(m_rotationStartPoint.getValue() - m_rotationCenter.getValue());
    //_pos[2] += m_radius * cos((m_angleStep - m_initAngle));
    //_pos[0] += m_radius * sin((m_angleStep - m_initAngle));
    p_position.setValue(_pos);

    // dV to compute circle tangente
    Vec3 _poskk;
    if (m_cameraUp.isSet() && m_cameraUp.getValue().norm() > 0.000001)
        _poskk = -cross(_pos-p_lookAt.getValue(),m_cameraUp.getValue());
    else
        _poskk = -cross(_pos-m_rotationCenter.getValue(),m_rotationAxis.getValue());
    //_poskk[2] += m_radius * cos((m_angleStep - m_initAngle+0.00001));
    //_poskk[0] += m_radius * sin((m_angleStep - m_initAngle+0.00001));

#ifdef my_debug
    std::cout << "totalTime: " << totalTime << std::endl;
    std::cout << "m_angleStep: " << m_angleStep << std::endl;
    std::cout << "_pos: " << _pos << std::endl;
    std::cout << "p_lookAt: " << p_lookAt.getValue() << std::endl;
#endif

    //Quat orientation  = getOrientationFromLookAt(_pos, p_lookAt.getValue());

    // Compute orientation
    Vec3 zAxis = -(p_lookAt.getValue() - _pos);
    Vec3 yAxis = zAxis.cross(_poskk);
    Vec3 xAxis = yAxis.cross(zAxis);
    xAxis.normalize();
    yAxis.normalize();
    zAxis.normalize();

#ifdef my_debug
    std::cout << "xAxis: " << xAxis << std::endl;
    std::cout << "yAxis: " << yAxis << std::endl;
    std::cout << "zAxis: " << zAxis << std::endl;
#endif

    Quat orientation  = Quat::createQuaterFromFrame(xAxis, yAxis, zAxis);
    orientation.normalize();

    p_orientation.setValue(orientation);

#ifdef my_debug
    //Quat orientation  = getOrientationFromLookAt(m_rotationStartPoint.getValue(), m_rotationCenter.getValue());
    //std::cout << "orientation: " << orientation << std::endl;
    Vec3 lookat = getLookAtFromOrientation(_pos, p_distance.getValue(), orientation);
    std::cout << "lookat: " << lookat << std::endl;
#endif

    return;
}


void RecordedCamera::moveCamera_translation()
{
    double simuTime = this->getContext()->getTime();
    SReal totalTime = this->m_endTime.getValue();
    simuTime -= m_startTime.getValue();
    totalTime -= m_startTime.getValue();

    if (totalTime == 0.0)
        totalTime = 200.0;

    if(m_translationPositions.isSet() && m_translationPositions.getValue().size() > 0)
    {
        unsigned int nbrPoints = (unsigned int)m_translationPositions.getValue().size();
        double timeBySegment = totalTime/(nbrPoints - 1);
        double simuTimeModTotalTime = fmod((SReal)simuTime,(SReal)totalTime);
        unsigned int currentIndexPoint = (unsigned int)floor((simuTimeModTotalTime/timeBySegment));
        double ratio = fmod(simuTimeModTotalTime,timeBySegment)/timeBySegment;

        // if the view up vector was not initialized
        if (m_cameraUp.getValue().norm() < 1e-6)
        this->initializeViewUp();

        if(currentIndexPoint < nbrPoints - 1)
        {
            Vec3 _pos = m_translationPositions.getValue()[currentIndexPoint];
            p_lookAt.setValue(m_translationPositions.getValue()[currentIndexPoint + 1]);
            Vec3 cameraFocal = p_lookAt.getValue() - _pos;

            // Set camera's position: linear interpolation
            p_position.setValue( m_translationPositions.getValue()[currentIndexPoint] + cameraFocal * ratio);

            // Set camera's orientation
            Vec3 zAxis = - (p_lookAt.getValue() - _pos);
            Vec3 xAxis = m_cameraUp.getValue().cross(zAxis);
            Vec3 yAxis = zAxis.cross(xAxis);    
            xAxis.normalize();
            yAxis.normalize();
            zAxis.normalize();
           
#ifdef my_debug
    std::cout << "xAxis: " << xAxis << std::endl;
    std::cout << "yAxis: " << yAxis << std::endl;
    std::cout << "zAxis: " << zAxis << std::endl;
#endif
         
            m_cameraUp.setValue(yAxis);
            Quat orientation  = Quat::createQuaterFromFrame(xAxis, yAxis, zAxis);
            orientation.normalize();
            p_orientation.setValue(orientation);
        }

        else if (currentIndexPoint == nbrPoints - 1 )
        {
            p_position.setValue(m_translationPositions.getValue()[currentIndexPoint]);
            p_lookAt.setValue(m_translationPositions.getValue()[currentIndexPoint]);
        }
    }

    return;
}


void RecordedCamera::handleEvent(sofa::core::objectmodel::Event *event)
{
    if (simulation::AnimateBeginEvent::checkEventType(event))
    {
        double simuTime = this->getContext()->getTime();
        double simuDT = this->getContext()->getDt();

        if (simuTime < m_nextStep)
            return;

        //std::cout << "rock & roll !" << std::endl;
        m_nextStep += simuDT;

        // init when start animation
       if(firstIterationforRotation & m_rotationMode.getValue())
            this->configureRotation();

        if(m_rotationMode.getValue())  
            this->moveCamera_rotation();

        if (firstIterationforTranslation & m_translationMode.getValue())
            this->configureTranslation();

        if(m_translationMode.getValue())
            this->moveCamera_translation();

        if(firstIterationforNavigation & m_navigationMode.getValue())
            this->configureNavigation();

        if(m_navigationMode.getValue())
            this->moveCamera_navigation();	
    }
    else if (sofa::core::objectmodel::KeypressedEvent::checkEventType(event))
    {
        sofa::core::objectmodel::KeypressedEvent* ke = static_cast<sofa::core::objectmodel::KeypressedEvent*>(event);
        cerr<<"RecordedCamera::handleEvent gets character " << ke->getKey() << endl;
    }

}


void RecordedCamera::configureRotation()
{
    // HACK: need to init again, as component init seems to be overwritten by viewer settings
    Vec3 _pos = m_rotationStartPoint.getValue();
    p_position.setValue(_pos);
    p_lookAt.setValue(m_rotationLookAt.getValue());
    p_distance.setValue((p_lookAt.getValue() - p_position.getValue()).norm());

    // dV to compute circle tangente
    Vec3 _poskk;
    if (m_cameraUp.isSet() && m_cameraUp.getValue().norm() > 0.000001)
        _poskk = -cross(_pos-p_lookAt.getValue(),m_cameraUp.getValue());
    else
        _poskk = -cross(_pos-m_rotationCenter.getValue(),m_rotationAxis.getValue());
    //_poskk[2] += m_radius * cos((m_angleStep - m_initAngle+0.00001));
    //_poskk[0] += m_radius * sin((m_angleStep - m_initAngle+0.00001));

#ifdef my_debug
    std::cout << "totalTime: " << totalTime << std::endl;
    std::cout << "m_angleStep: " << m_angleStep << std::endl;
    std::cout << "_pos: " << _pos << std::endl;
    std::cout << "p_lookAt: " << p_lookAt.getValue() << std::endl;
#endif

    //Quat orientation  = getOrientationFromLookAt(_pos, p_lookAt.getValue());

    // Compute orientation
    Vec3 zAxis = -(p_lookAt.getValue() - _pos);
    Vec3 yAxis = zAxis.cross(_poskk);
    Vec3 xAxis = yAxis.cross(zAxis);
    xAxis.normalize();
    yAxis.normalize();
    zAxis.normalize();

#ifdef my_debug
    std::cout << "xAxis: " << xAxis << std::endl;
    std::cout << "yAxis: " << yAxis << std::endl;
    std::cout << "zAxis: " << zAxis << std::endl;
#endif

    Quat orientation  = Quat::createQuaterFromFrame(xAxis, yAxis, zAxis);
    orientation.normalize();

    p_orientation.setValue(orientation);

    // Compute rotation settings: radius and init angle
    /*
        m_radius = (m_rotationCenter.getValue() - m_rotationStartPoint.getValue()).norm();

        Vec3 _pos = p_position.getValue();
        if (_pos[0]>=0)
            m_initAngle = asin(_pos[2]/m_radius);
        else
            m_initAngle = M_PI - asin(_pos[2]/m_radius);
    */
#ifdef my_debug
    std::cout << "m_rotationStartPoint: " << m_rotationStartPoint << std::endl;
    std::cout << "m_rotationCenter: " << m_rotationCenter << std::endl;
    std::cout << "m_rotationSpeed: " << m_rotationSpeed << std::endl;
    //std::cout << "init p_lookAt: " << p_lookAt << std::endl;
    //std::cout << "m_initAngle: " << m_initAngle << std::endl;
#endif

    firstIterationforRotation = false;

    return;
}

void RecordedCamera::configureTranslation()
{
    if(m_translationPositions.isSet() && m_translationPositions.getValue().size() > 1)
    {
        // Set camera's position
        p_position.setValue(m_translationPositions.getValue()[0]);
        p_lookAt.setValue(m_translationPositions.getValue()[1]);

        // Set camera's orientation
        this->initializeViewUp();
        Vec3 zAxis = - m_translationPositions.getValue()[1] +  m_translationPositions.getValue()[0];
        Vec3 yAxis = m_cameraUp.getValue();
        Vec3 xAxis = yAxis.cross(zAxis);
        xAxis.normalize();
        yAxis.normalize();
        zAxis.normalize();
        Quat orientation  = Quat::createQuaterFromFrame(xAxis, yAxis, zAxis);
        orientation.normalize();
        p_orientation.setValue(orientation);

        firstIterationforTranslation = false;
    }
    return;
}

void RecordedCamera:: configureNavigation()
{
    if(m_translationPositions.getValue().size() > 1 &&  m_translationOrientations.getValue().size() == m_translationPositions.getValue().size())
    {
        // Set camera's position
        p_position.setValue(m_translationPositions.getValue()[0]);

        // Set camera's orientation
        p_orientation.setValue(m_translationOrientations.getValue()[0]);

        firstIterationforNavigation = false;
    }
    return;
}

void RecordedCamera::initializeViewUp()
{
    if(m_translationPositions.isSet() && m_translationPositions.getValue().size() > 1)
    {
        Vec3 zAxis = m_translationPositions.getValue()[1] -  m_translationPositions.getValue()[0];
        zAxis.normalize();
        Vec3 xRef(1,0,0); 
        // Initialize the view-up vector with the reference vector the "most perpendicular" to zAxis.
         m_cameraUp.setValue(xRef);
        double normCrossProduct = cross(zAxis,xRef).norm();
        for(int i = 1; i<3; ++ i)
        {
            Vec3 vecRef(0,0,0);
            vecRef[i] = 1;
            if(cross(zAxis,vecRef).norm() >= normCrossProduct )
            {
                normCrossProduct = cross(zAxis,vecRef).norm();
                m_cameraUp.setValue(vecRef);
            }
        }
    }
}

void RecordedCamera::manageEvent(core::objectmodel::Event* e)
{
    if(p_activated.getValue())
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
    int wheelDelta = me->getWheelDelta();

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

    p_position.endEdit();

}


void RecordedCamera::moveCamera_mouse(int x, int y)
{
    Quat newQuat;
    const unsigned int widthViewport = p_widthViewport.getValue();
    const unsigned int heightViewport = p_heightViewport.getValue();

    if (isMoving)
    {
        if (currentMode == TRACKBALL_MODE)
        {
            float x1 = (2.0f * widthViewport / 2.0f - widthViewport) / widthViewport;
            float y1 = (heightViewport- 2.0f *heightViewport / 2.0f) /heightViewport;
            float x2 = (2.0f * (x + (-lastMousePosX + widthViewport / 2.0f)) - widthViewport) / widthViewport;
            float y2 = (heightViewport- 2.0f * (y + (-lastMousePosY +heightViewport / 2.0f))) /heightViewport;
            //std::cout << x1 << " " << y1 << " " << x2 << " " << y2 << std::endl;
            currentTrackball.ComputeQuaternion(x1, y1, x2, y2);
            //fetch rotation
            newQuat = currentTrackball.GetQuaternion();
            Vec3 pivot;
            switch (p_pivot.getValue())
            {
            case WORLD_CENTER_PIVOT:
                pivot = Vec3(0.0, 0.0, 0.0);
                break;
            case SCENE_CENTER_PIVOT :
            default:
                pivot = sceneCenter;
                break;
            }
            //pivot = p_lookAt.getValue();
            //pivot = (Vec3(x,y, p_distance.getValue()));
            //std::cout << "Pivot : " <<  pivot << std::endl;
            //rotateCameraAroundPoint(newQuat, pivot);


            BaseCamera::rotateWorldAroundPoint(newQuat, pivot, this->getOrientation());
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


void RecordedCamera::drawRotation()
{
    Vec3 _pos = m_rotationStartPoint.getValue();
    //Vec3 _center = m_rotationCenter.getValue();

    //double _initAngle = 0.0;

    // Compute rotation settings: radius and init angle
    /*
        m_radius = (_center - _pos).norm();

        if (_pos[0]>=0)
            _initAngle = asin(_pos[2]/m_radius);
        else
            _initAngle = M_PI - asin(_pos[2]/m_radius);
    */

    m_rotationPoints.resize(100);
    double _angleStep = 2*M_PI/100;
    for (unsigned int i = 0; i<100; ++i)
    {
        // Compute cartesian coordinates from cylindrical ones
        _pos = m_rotationCenter.getValue();
        //_pos[2] += m_radius * cos((_angleStep*i - _initAngle));
        //_pos[0] += m_radius * sin((_angleStep*i - _initAngle));
        helper::Quater<double> q(m_rotationAxis.getValue(), _angleStep*i);
        _pos += q.rotate(m_rotationStartPoint.getValue() - m_rotationCenter.getValue());
        m_rotationPoints[i] = _pos;
    }

    return;
}

void RecordedCamera::draw(const core::visual::VisualParams* /*vparams*/)
{
#ifndef SOFA_NO_OPENGL
    // Draw rotation path
    if(p_drawRotation.getValue())
    {
        if (m_rotationPoints.empty())
            return;

        glDisable(GL_LIGHTING);
        glColor3f(0,1,0.5);

        // Camera positions
        glBegin(GL_LINES);
        for (unsigned int i=0; i<m_rotationPoints.size()-1; ++i)
        {
            glVertex3f((float)m_rotationPoints[i  ][0], (float)m_rotationPoints[i  ][1], (float)m_rotationPoints[i  ][2]);
            glVertex3f((float)m_rotationPoints[i+1][0], (float)m_rotationPoints[i+1][1], (float)m_rotationPoints[i+1][2]);
        }
        glVertex3f((float)m_rotationPoints.back()[0], (float)m_rotationPoints.back()[1], (float)m_rotationPoints.back()[2]);
        glVertex3f((float)m_rotationPoints[0    ][0], (float)m_rotationPoints[0    ][1], (float)m_rotationPoints[0    ][2]);
        glEnd();

        Vec3 _lookAt = m_rotationLookAt.getValue();
        unsigned int dx = 4;
        std::size_t ratio = m_rotationPoints.size()/dx;
        glBegin(GL_LINES);
        for (unsigned int i=0; i<dx; ++i)
        {
            glVertex3f((float)m_rotationPoints[i*ratio][0], (float)m_rotationPoints[i*ratio][1], (float)m_rotationPoints[i*ratio][2]);
            glVertex3f((float)_lookAt[0], (float)_lookAt[1], (float)_lookAt[2]);
        }
        glEnd();
    }

    // Draw translation path
    if(p_drawTranslation.getValue())
    {
        if (m_translationPositions.getValue().size() < 2)
            return;

        glDisable(GL_LIGHTING);
        glColor3f(0,1,0.5);

        // Camera positions
        glBegin(GL_LINES);
        helper::vector <Vec3> _positions = m_translationPositions.getValue();
        for (unsigned int i=0; i < _positions.size()-1; ++i)
        {
            glVertex3f((float)_positions[i  ][0], (float)_positions[i  ][1], (float)_positions[i  ][2]);
            glVertex3f((float)_positions[i+1][0], (float)_positions[i+1][1], (float)_positions[i+1][2]);
        }

        glEnd ();
    }
#endif /* SOFA_NO_OPENGL */
}

} // namespace visualmodel

} // namespace component

} // namespace sofa
