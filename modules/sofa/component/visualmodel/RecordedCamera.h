#ifndef RECORDEDCAMERA_H
#define RECORDEDCAMERA_H

#include <sofa/component/visualmodel/BaseCamera.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{
#define PI 3.14159265

class SOFA_OPENGL_VISUAL_API RecordedCamera : public BaseCamera
{
public:
    SOFA_CLASS(RecordedCamera, BaseCamera);

    typedef BaseCamera::Vec3 Vec3;
    typedef BaseCamera::Quat Quat;
protected:
    RecordedCamera();
    virtual ~RecordedCamera() {}
public:
    virtual void init();

    virtual void reinit();

    virtual void reset();

    virtual void handleEvent(sofa::core::objectmodel::Event *);

    //virtual void rotateWorldAroundPoint(Quat &rotation, const Vec3 &point);

    enum  { TRACKBALL_MODE, PAN_MODE, ZOOM_MODE, WHEEL_ZOOM_MODE, NONE_MODE };
    enum  { SCENE_CENTER_PIVOT = 0, WORLD_CENTER_PIVOT = 1};

    Data<double> p_zoomSpeed;
    Data<double> p_panSpeed;
    Data<int> p_pivot;

    void draw(const core::visual::VisualParams* vparams);

private:
    int currentMode;
    bool isMoving;
    int lastMousePosX, lastMousePosY;
    helper::gl::Trackball currentTrackball;

    void moveCamera_rotation();

    // Kepp functions for mouse interaction (TODO: removed them and allow interactive and recorded camera in same scene)
    void moveCamera_mouse(int x, int y);
    void manageEvent(core::objectmodel::Event* e);
    void processMouseEvent(core::objectmodel::MouseEvent* me);

    void configureRotation();
    void drawRotation();

public:
    Data<SReal> m_startTime;
    Data<SReal> m_endTime;

    Data <bool> m_rotationMode;
    Data <SReal> m_rotationSpeed;
    Data <Vec3> m_rotationCenter;
    Data <Vec3> m_rotationStartPoint;
    Data <Vec3> m_rotationLookAt;
    Data <Vec3> m_rotationAxis;
    Data <Vec3> m_cameraUp;

    Data <bool> p_drawRotation;

    //Data <SReal> m_translationSpeed;
    //Data <sofa::helper::vector<Vec3> > m_translationPositions;
    //Data <sofa::helper::vector<Vec3> > m_translationOrientations;

protected:
    double m_nextStep;
    double m_angleStep;
    //double m_initAngle;
    //double m_radius;
    bool firstIteration;

    sofa::helper::vector <Vec3> m_rotationPoints;
};

} // namespace visualmodel

} // namespace component

} // namespace sofa

#endif // RecordedCamera_H
