#ifndef SOFA_COMPONENT_VISUALMODEL_INTERACTIVECAMERA_H
#define SOFA_COMPONENT_VISUALMODEL_INTERACTIVECAMERA_H

#include <sofa/component/visualmodel/BaseCamera.h>
#include <sofa/component/component.h>
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
    enum  { SCENE_CENTER_PIVOT = 0, WORLD_CENTER_PIVOT = 1};

    Data<double> p_zoomSpeed;
    Data<double> p_panSpeed;
    Data<int> p_pivot;

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

#endif // SOFA_COMPONENT_VISUALMODEL_INTERACTIVECAMERA_H
