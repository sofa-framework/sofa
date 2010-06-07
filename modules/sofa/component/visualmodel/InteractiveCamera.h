#ifndef INTERACTIVECAMERA_H
#define INTERACTIVECAMERA_H

#include <sofa/component/visualmodel/BaseCamera.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

class SOFA_COMPONENT_VISUALMODEL_API InteractiveCamera : public BaseCamera
{
public:
    SOFA_CLASS(InteractiveCamera, BaseCamera);

    enum  { TRACKBALL_MODE, PAN_MODE, ZOOM_MODE, WHEEL_ZOOM_MODE, NONE_MODE };

    Data<double> p_zoomSpeed;
    Data<double> p_panSpeed;

    InteractiveCamera();
    virtual ~InteractiveCamera();

private:
    int currentMode;
    bool isMoving;
    int lastMousePosX, lastMousePosY;
    helper::gl::Trackball currentTrackball;

    void internalUpdate();
    void moveCamera(int x, int y);
    void manageEvent(core::objectmodel::Event* e);
    void processMouseEvent(core::objectmodel::MouseEvent* me);
    void processKeyPressedEvent(core::objectmodel::KeypressedEvent* kpe);
    void processKeyReleasedEvent(core::objectmodel::KeyreleasedEvent* kre);
};

} // namespace visualmodel

} // namespace component

} // namespace sofa

#endif // INTERACTIVECAMERA_H
