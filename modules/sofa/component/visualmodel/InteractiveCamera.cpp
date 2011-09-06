#include <sofa/component/visualmodel/InteractiveCamera.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

SOFA_DECL_CLASS(InteractiveCamera)

int InteractiveCameraClass = core::RegisterObject("InteractiveCamera")
        .add< InteractiveCamera >()
        .addAlias("Camera")
        ;


InteractiveCamera::InteractiveCamera()
    :p_zoomSpeed(initData(&p_zoomSpeed, (double) 250.0 , "zoomSpeed", "Zoom Speed"))
    ,p_panSpeed(initData(&p_panSpeed, (double) 0.1 , "panSpeed", "Pan Speed"))
    ,p_pivot(initData(&p_pivot, 0 , "pivot", "Pivot (0 => Scene center, 1 => World Center"))
    ,currentMode(InteractiveCamera::NONE_MODE)
    ,isMoving(false)
{
}

InteractiveCamera::~InteractiveCamera()
{
}

void InteractiveCamera::internalUpdate()
{
    //TODO: update pan and zoom speeds
}

void InteractiveCamera::moveCamera(int x, int y)
{
    float x1, x2, y1, y2;
    Quat newQuat;
    const unsigned int widthViewport = p_widthViewport.getValue();
    const unsigned int heightViewport = p_heightViewport.getValue();

    //std::cout << "widthViewport: " << widthViewport << std::endl;
    //std::cout << "heightViewport: " << heightViewport << std::endl;
    //std::cout << "x: " << x << std::endl;
    //std::cout << "y: " << y << std::endl;

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
            rotateWorldAroundPoint(newQuat, pivot, this->getOrientation());
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


void InteractiveCamera::manageEvent(core::objectmodel::Event* e)
{
    core::objectmodel::MouseEvent* me;
    core::objectmodel::KeypressedEvent* kpe;
    core::objectmodel::KeyreleasedEvent* kre;

    if(p_activated.getValue())
    {
        //Dispatch event
        if ((me = dynamic_cast<core::objectmodel::MouseEvent* > (e)))
            processMouseEvent(me);
        else if ((kpe = dynamic_cast<core::objectmodel::KeypressedEvent* > (e)))
            processKeyPressedEvent(kpe);
        else if ((kre = dynamic_cast<core::objectmodel::KeyreleasedEvent* > (e)))
            processKeyReleasedEvent(kre);

        internalUpdate();
    }
    else
    {
        isMoving = false;
        currentMode = NONE_MODE;
    }
}

void InteractiveCamera::processMouseEvent(core::objectmodel::MouseEvent* me)
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

    moveCamera(posX, posY);

    p_position.endEdit();

}

void InteractiveCamera::processKeyPressedEvent(core::objectmodel::KeypressedEvent* kpe)
{
    char keyPressed = kpe->getKey();

    switch(keyPressed)
    {
    case 'a':
    case 'A':
    {
        //glPushMatrix();
        //glLoadIdentity();
        //helper::gl::Axis(p_position.getValue(), p_orientation.getValue(), 10.0);
        //glPopMatrix();
        break;
    }
    default:
    {
        break;
    }
    }
}

void InteractiveCamera::processKeyReleasedEvent(core::objectmodel::KeyreleasedEvent* /* kre */)
{

}

} // namespace visualmodel

} // namespace component

} // namespace sofa
