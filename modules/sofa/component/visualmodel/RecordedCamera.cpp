#include <sofa/component/visualmodel/RecordedCamera.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/common/AnimateBeginEvent.h>
#include <sofa/simulation/common/AnimateEndEvent.h>

#include <sofa/helper/system/glut.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

SOFA_DECL_CLASS(RecordedCamera)

int RecordedCameraClass = core::RegisterObject("RecordedCamera")
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
    , m_rotationMode(initData(&m_rotationMode, (bool)true , "rotationMode", "If true, rotation will be performed"))
    , m_rotationSpeed(initData(&m_rotationSpeed, (SReal)0.1 , "rotationSpeed", "rotation Speed"))
    , m_rotationCenter(initData(&m_rotationCenter, "rotationCenter", "Rotation center coordinates"))
    , m_rotationStartPoint(initData(&m_rotationStartPoint, "rotationStartPoint", "Rotation start position coordinates"))
    , m_rotationLookAt(initData(&m_rotationLookAt, "rotationLookAt", "Position to be focused during rotation"))
    , p_drawRotation(initData(&p_drawRotation, (bool)false , "drawRotation", "If true, will draw the rotation path"))
    //, m_translationSpeed(initData(&m_translationSpeed, (SReal)0.1 , "translationSpeed", "Pan Speed"))
    //, m_translationPositions(initData(&m_translationPositions, "translationPositions", "Pan Speed"))
    //, m_translationOrientations(initData(&m_translationOrientations, "translationOrientations", "Pan Speed"))
    , m_nextStep(0.0)
    , m_angleStep(0.0)
    , m_initAngle(0.0)
    ,firstIteration(true)
{
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


void RecordedCamera::moveCamera_rotation()
{
    // Compute angle from Dt
    double simuTime = this->getContext()->getTime();
    //double simuDT = this->getContext()->getDt();
    SReal totalTime = this->m_endTime.getValue();
    simuTime -= m_startTime.getValue();

    if (totalTime == 0.0)
        totalTime = 200.0;

    double ratio = (simuTime / totalTime);
    m_angleStep = 2*PI * ratio;

    // Compute cartesian coordinates from cylindrical ones
    Vec3 _pos = m_rotationCenter.getValue();
    _pos[2] += m_radius * cos((m_angleStep - m_initAngle));
    _pos[0] += m_radius * sin((m_angleStep - m_initAngle));
    p_position.setValue(_pos);

    // Compute coordinate of point + dV to compute circle tangente
    Vec3 _poskk = m_rotationCenter.getValue();
    _poskk[2] += m_radius * cos((m_angleStep - m_initAngle+0.00001));
    _poskk[0] += m_radius * sin((m_angleStep - m_initAngle+0.00001));

#ifdef my_debug
    std::cout << "totalTime: " << totalTime << std::endl;
    std::cout << "m_angleStep: " << m_angleStep << std::endl;
    std::cout << "_pos: " << _pos << std::endl;
    std::cout << "p_lookAt: " << p_lookAt.getValue() << std::endl;
#endif

    //Quat orientation  = getOrientationFromLookAt(_pos, p_lookAt.getValue());

    // Compute orientation
    Vec3 zAxis = -(p_lookAt.getValue() - _pos);
    zAxis.normalize();

    Vec3 xAxis = (_poskk - _pos); xAxis.normalize();
    xAxis.normalize();

    Vec3 yAxis = zAxis.cross(xAxis);

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



void RecordedCamera::handleEvent(sofa::core::objectmodel::Event *event)
{
    if (dynamic_cast<simulation::AnimateBeginEvent*>(event))
    {
        double simuTime = this->getContext()->getTime();
        double simuDT = this->getContext()->getDt();

        if (simuTime < m_nextStep)
            return;

        //std::cout << "rock & roll !" << std::endl;
        m_nextStep += simuDT;

        // init when start animation
        if(firstIteration & m_rotationMode.getValue())
            this->configureRotation();


        this->moveCamera_rotation();
    }
}


void RecordedCamera::configureRotation()
{
    // HACK: need to init again, as component init seems to be overwritten by viewer settings
    p_position.setValue(m_rotationStartPoint.getValue());
    p_lookAt.setValue(m_rotationLookAt.getValue());
    p_distance.setValue((p_lookAt.getValue() - p_position.getValue()).norm());


    // Compute rotation settings: radius and init angle
    m_radius = (m_rotationCenter.getValue() - m_rotationStartPoint.getValue()).norm();

    Vec3 _pos = p_position.getValue();
    if (_pos[0]>=0)
        m_initAngle = asin(_pos[2]/m_radius);
    else
        m_initAngle = PI - asin(_pos[2]/m_radius);

#ifdef my_debug
    std::cout << "m_rotationStartPoint: " << m_rotationStartPoint << std::endl;
    std::cout << "m_rotationCenter: " << m_rotationCenter << std::endl;
    std::cout << "m_rotationSpeed: " << m_rotationSpeed << std::endl;
    std::cout << "init p_lookAt: " << p_lookAt << std::endl;
    std::cout << "m_initAngle: " << m_initAngle << std::endl;
#endif

    firstIteration = false;

    return;
}


void RecordedCamera::manageEvent(core::objectmodel::Event* e)
{

    core::objectmodel::MouseEvent* me;
    //core::objectmodel::KeypressedEvent* kpe;
    //core::objectmodel::KeyreleasedEvent* kre;

    if(p_activated.getValue())
    {
        //Dispatch event
        if ((me = dynamic_cast<core::objectmodel::MouseEvent* > (e)))
            processMouseEvent(me);
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
    Vec3 _center = m_rotationCenter.getValue();

    double _initAngle = 0.0;

    // Compute rotation settings: radius and init angle
    m_radius = (_center - _pos).norm();

    if (_pos[0]>=0)
        _initAngle = asin(_pos[2]/m_radius);
    else
        _initAngle = PI - asin(_pos[2]/m_radius);

    m_rotationPoints.resize(100);
    double _angleStep = 2*PI/100;
    for (unsigned int i = 0; i<100; ++i)
    {
        // Compute cartesian coordinates from cylindrical ones
        _pos = m_rotationCenter.getValue();
        _pos[2] += m_radius * cos((_angleStep*i - _initAngle));
        _pos[0] += m_radius * sin((_angleStep*i - _initAngle));
        m_rotationPoints[i] = _pos;
    }

    return;
}


void RecordedCamera::draw(const core::visual::VisualParams* vparams)
{
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
            glVertex3f(m_rotationPoints[i][0], m_rotationPoints[i][1], m_rotationPoints[i][2]);
            glVertex3f(m_rotationPoints[i+1][0], m_rotationPoints[i+1][1], m_rotationPoints[i+1][2]);
        }
        glVertex3f(m_rotationPoints.back()[0], m_rotationPoints.back()[1], m_rotationPoints.back()[2]);
        glVertex3f(m_rotationPoints[0][0], m_rotationPoints[0][1], m_rotationPoints[0][2]);
        glEnd();

        Vec3 _lookAt = m_rotationLookAt.getValue();
        unsigned int dx = 4;
        unsigned int ratio = m_rotationPoints.size()/dx;
        glBegin(GL_LINES);
        for (unsigned int i=0; i<dx; ++i)
        {
            glVertex3f(m_rotationPoints[i*ratio][0], m_rotationPoints[i*ratio][1], m_rotationPoints[i*ratio][2]);
            glVertex3f(_lookAt[0], _lookAt[1], _lookAt[2]);
        }
        glEnd();
    }
}


} // namespace visualmodel

} // namespace component

} // namespace sofa
