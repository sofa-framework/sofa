/******************************************************************************
 *       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
 *                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
 *                                                                             *
 * This program is free software; you can redistribute it and/or modify it     *
 * under the terms of the GNU General Public License as published by the Free  *
 * Software Foundation; either version 2 of the License, or (at your option)   *
 * any later version.                                                          *
 *                                                                             *
 * This program is distributed in the hope that it will be useful, but WITHOUT *
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
 * more details.                                                               *
 *                                                                             *
 * You should have received a copy of the GNU General Public License along     *
 * with this program; if not, write to the Free Software Foundation, Inc., 51  *
 * Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
 *******************************************************************************
 *                            SOFA :: Applications                             *
 *                                                                             *
 * Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
 * H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
 * M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
 *                                                                             *
 * Contact information: contact@sofa-framework.org                             *
 ******************************************************************************/
#ifndef SOFA_VIEWER_H
#define SOFA_VIEWER_H

#include <qstring.h>
#include <qwidget.h>
#include <sofa/helper/Factory.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/CollisionModel.h>
#ifdef SOFA_QT4
#include <QEvent>
#include <QMouseEvent>
#include <QKeyEvent>
#include <QTabWidget>
#include <QTimer>
#else
#include <qevent.h>
#include <qtabwidget.h>
#include <qtimer.h>
#endif
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/gui/PickHandler.h>

#include <qcursor.h>

#include <sofa/helper/gl/Capture.h>
#include <sofa/gui/qt/SofaVideoRecorderManager.h>
#ifdef SOFA_HAVE_FFMPEG
#include <sofa/helper/gl/VideoRecorder.h>
#endif //SOFA_HAVE_FFMPEG

#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <sofa/core/objectmodel/MouseEvent.h>
#include <sofa/core/collision/Pipeline.h>

#include <sofa/component/configurationsetting/ViewerSetting.h>

//instruments handling
#include <sofa/component/controller/Controller.h>
#include <sofa/defaulttype/LaparoscopicRigidTypes.h>
//#include <sofa/simulation/common/GrabVisitor.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/simulation/common/UpdateMappingVisitor.h>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/component/visualmodel/InteractiveCamera.h>

#ifdef SOFA_QT4
#include <QEvent>
#include <QMouseEvent>
#include <QKeyEvent>
#else
#include <qevent.h>
#endif

namespace sofa
{

namespace gui
{

namespace qt
{

namespace viewer
{

using namespace sofa::defaulttype;
enum
{
    BTLEFT_MODE = 101, BTRIGHT_MODE = 102, BTMIDDLE_MODE = 103,
};


class SofaViewer
{

public:



    SofaViewer()
        : groot(NULL)
        , m_isControlPressed(false)
        , _video(false)
        , _shadow(false)
        ,_gl_shadow(false)
        ,_axis(false)
        ,backgroundColour(Vector3())
        ,backgroundImage("textures/SOFA_logo.bmp"), ambientColour(Vector3())
        ,currentCamera(NULL)
    {
    }

    virtual ~SofaViewer()
    {
    }

    virtual void drawColourPicking (core::CollisionModel::ColourCode /*code*/) {};



    virtual QWidget* getQWidget()=0;

    virtual sofa::simulation::Node* getScene()
    {
        return groot;
    }
    virtual const std::string& getSceneFileName()
    {
        return sceneFileName;
    }
    virtual void setSceneFileName(const std::string &f)
    {
        sceneFileName = f;
    }
    ;

    virtual void setCameraMode(component::visualmodel::BaseCamera::CameraType mode)
    {
        currentCamera->setCameraType(mode);
    }

    virtual void setScene(sofa::simulation::Node* scene, const char* filename =
            NULL, bool /*keepParams*/= false)
    {

        std::string file =
            filename ? sofa::helper::system::SetDirectory::GetFileName(
                    filename) : std::string();
        std::string
        screenshotPrefix =
            sofa::helper::system::SetDirectory::GetParentDir(
                    sofa::helper::system::DataRepository.getFirstPath().c_str())
            + std::string("/share/screenshots/") + file
            + std::string("_");
        capture.setPrefix(screenshotPrefix);
#ifdef SOFA_HAVE_FFMPEG
        videoRecorder.setPrefix(screenshotPrefix);
#endif //SOFA_HAVE_FFMPEG
        sceneFileName = filename ? filename : std::string("default.scn");
        groot = scene;
        initTexturesDone = false;
        sceneBBoxIsValid = false;


        //Camera initialization
        if (groot)
        {
            groot->get(currentCamera);
            if (!currentCamera)
            {
                currentCamera = new component::visualmodel::InteractiveCamera();
                groot->addObject(currentCamera);
                //std::cout << "Create Default Camera" << std::endl;
            }
            sofa::defaulttype::Vec3d minBBox, maxBBox;

            sofa::simulation::getSimulation()->computeBBox(groot, minBBox.ptr(),maxBBox.ptr());

            currentCamera->setBoundingBox(minBBox, maxBBox);

        }

        pick.init();
        //if (!keepParams) resetView();
    }

    virtual QString helpString()=0;
    virtual bool ready()
    {
        return true;
    }
    virtual void wait()
    {
    }

    //Allow to configure your viewer using the Sofa Component, ViewerSetting
    virtual void configure(sofa::component::configurationsetting::ViewerSetting* viewerConf)
    {
        if (viewerConf->getCameraModeId() == component::visualmodel::BaseCamera::ORTHOGRAPHIC_TYPE)
            setCameraMode(component::visualmodel::BaseCamera::ORTHOGRAPHIC_TYPE);
        else
            setCameraMode(component::visualmodel::BaseCamera::PERSPECTIVE_TYPE);
        if ( viewerConf->getPickingMethodId() == gui::PickHandler::RAY_CASTING)
            pick.setPickingMethod( gui::PickHandler::RAY_CASTING );
        else
            pick.setPickingMethod( gui::PickHandler::SELECTION_BUFFER);
    }

    //Fonctions needed to take a screenshot
    virtual const std::string screenshotName()
    {
        return capture.findFilename().c_str();
    }
    ;
    virtual void setPrefix(const std::string filename)
    {
        capture.setPrefix(filename);
    }
    ;
    virtual void screenshot(const std::string filename, int compression_level =
            -1)
    {
        capture.saveScreen(filename, compression_level);
    }

    virtual void getView(Vec3d& pos, Quat& ori) const
    {
        if (!currentCamera)
            return;

        const Vec3d& camPosition = currentCamera->getPosition();
        const Quat& camOrientation = currentCamera->getOrientation();

        pos[0] = camPosition[0];
        pos[1] = camPosition[1];
        pos[2] = camPosition[2];

        ori[0] = camOrientation[0];
        ori[1] = camOrientation[1];
        ori[2] = camOrientation[2];
        ori[3] = camOrientation[3];
    }

    virtual void setView(const Vec3d& pos, const Quat &ori)
    {
        Vec3d position;
        Quat orientation;
        for (unsigned int i=0 ; i<3 ; i++)
        {
            position[i] = pos[i];
            orientation[i] = ori[i];
        }
        orientation[3] = ori[3];

        if (currentCamera)
            currentCamera->setView(position, orientation);

        getQWidget()->update();
    }

    virtual void moveView(const Vec3d& pos, const Quat &ori)
    {
        if (!currentCamera)
            return;

        currentCamera->moveCamera(pos, ori);
        getQWidget()->update();
    }

    virtual void newView()
    {
        if (!currentCamera || !groot)
            return;

        currentCamera->setDefaultView(groot->getGravityInWorld());
    }

    virtual void resetView()
    {
        getQWidget()->update();
    }

    virtual void removeViewerTab(QTabWidget *)
    {
    }
    ;
    virtual void configureViewerTab(QTabWidget *)
    {
    }
    ;

    virtual void setBackgroundColour(float r, float g, float b)
    {
        _background = 2;
        backgroundColour[0] = r;
        backgroundColour[1] = g;
        backgroundColour[2] = b;
    }

    virtual void setBackgroundImage(std::string imageFileName)
    {
        _background = 0;
        backgroundImage = imageFileName;
    }

    std::string getBackgroundImage()
    {
        return backgroundImage;
    }
    PickHandler* getPickHandler()
    {
        return &pick;
    }

protected:

    // ---------------------- Here are the Keyboard controls   ----------------------
    void keyPressEvent(QKeyEvent * e)
    {
        sofa::core::objectmodel::KeypressedEvent kpe(e->key());
        currentCamera->manageEvent(&kpe);

        switch (e->key())
        {
        case Qt::Key_T:
        {
            if (currentCamera->getCameraType() == component::visualmodel::BaseCamera::ORTHOGRAPHIC_TYPE)
                setCameraMode(component::visualmodel::BaseCamera::PERSPECTIVE_TYPE);
            else
                setCameraMode(component::visualmodel::BaseCamera::ORTHOGRAPHIC_TYPE);
            break;
        }
        case Qt::Key_Shift:
            GLint viewport[4];
            glGetIntegerv(GL_VIEWPORT,viewport);
            pick.activateRay(viewport[2],viewport[3]);
            break;
        case Qt::Key_B:
            // --- change background
        {
            _background = (_background + 1) % 3;
            break;
        }
        case Qt::Key_L:
            // --- draw shadows
        {
            if (_gl_shadow)
                _shadow = !_shadow;
            break;
        }
        case Qt::Key_R:
            // --- draw axis
        {
            _axis = !_axis;
            break;
        }
        case Qt::Key_S:
        {
            screenshot(capture.findFilename());
            break;
        }
        case Qt::Key_V:
            // --- save video
        {
            if(!_video)
            {
                switch (SofaVideoRecorderManager::getInstance()->getRecordingType())
                {
                case SofaVideoRecorderManager::SCREENSHOTS :
                    break;
                case SofaVideoRecorderManager::MOVIE :
                {
#ifdef SOFA_HAVE_FFMPEG
                    SofaVideoRecorderManager* videoManager = SofaVideoRecorderManager::getInstance();
                    unsigned int framerate = videoManager->getFramerate(); //img.s-1
                    unsigned int freq = ceil(1000/framerate); //ms
                    unsigned int bitrate = videoManager->getBitrate();
                    std::string videoFilename = videoRecorder.findFilename(videoManager->getCodecExtension());
                    videoRecorder.init( videoFilename, framerate, bitrate);
                    captureTimer.start(freq);
#endif

                    break;
                }
                default :
                    break;
                }

            }
            else
            {
                switch (SofaVideoRecorderManager::getInstance()->getRecordingType())
                {
                case SofaVideoRecorderManager::SCREENSHOTS :
                    break;
                case SofaVideoRecorderManager::MOVIE :
                {
                    captureTimer.stop();
#ifdef SOFA_HAVE_FFMPEG
                    videoRecorder.finishVideo();
#endif //SOFA_HAVE_FFMPEG
                    break;
                }
                default :
                    break;
                }
            }

            _video = !_video;
            //capture.setCounter();

            break;
        }
        case Qt::Key_W:
            // --- save current view
        {
            saveView();
            break;
        }
        case Qt::Key_Control:
        {
            m_isControlPressed = true;
            //cerr<<"QtViewer::keyPressEvent, CONTROL pressed"<<endl;
            break;
        }
        default:
        {
            e->ignore();
        }
        }
    }

    void keyReleaseEvent(QKeyEvent * e)
    {
        sofa::core::objectmodel::KeyreleasedEvent kre(e->key());
        currentCamera->manageEvent(&kre);

        switch (e->key())
        {
        case Qt::Key_Shift:
            pick.deactivateRay();

            break;
        case Qt::Key_Control:
        {
            m_isControlPressed = false;

            // Send Control Release Info to a potential ArticulatedRigid Instrument
            sofa::core::objectmodel::MouseEvent mouseEvent(
                sofa::core::objectmodel::MouseEvent::Reset);
            if (groot)
                groot->propagateEvent(&mouseEvent);
        }
        default:
        {
            e->ignore();
        }
        }

        if (isControlPressed())
        {
            sofa::core::objectmodel::KeyreleasedEvent keyEvent(e->key());
            if (groot)
                groot->propagateEvent(&keyEvent);
        }
    }

    bool isControlPressed() const
    {
        return m_isControlPressed;
    }

    // ---------------------- Here are the Mouse controls   ----------------------
    void wheelEvent(QWheelEvent *e)
    {
        //<CAMERA API>
        sofa::core::objectmodel::MouseEvent me(sofa::core::objectmodel::MouseEvent::Wheel,e->delta());
        currentCamera->manageEvent(&me);

        getQWidget()->update();
    }

    void mouseMoveEvent ( QMouseEvent *e )
    {
        //<CAMERA API>
        sofa::core::objectmodel::MouseEvent me(sofa::core::objectmodel::MouseEvent::Move,e->x(), e->y());
        currentCamera->manageEvent(&me);

        getQWidget()->update();
    }

    void mousePressEvent ( QMouseEvent * e)
    {
        //<CAMERA API>
        sofa::core::objectmodel::MouseEvent* mEvent = NULL;
        if (e->button() == Qt::LeftButton)
            mEvent = new sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::LeftPressed, e->x(), e->y());
        else if (e->button() == Qt::RightButton)
            mEvent = new sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::RightPressed, e->x(), e->y());
        else if (e->button() == Qt::MidButton)
            mEvent = new sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::MiddlePressed, e->x(), e->y());
        currentCamera->manageEvent(mEvent);

        getQWidget()->update();
    }

    void mouseReleaseEvent ( QMouseEvent * e)
    {
        //<CAMERA API>
        sofa::core::objectmodel::MouseEvent* mEvent = NULL;
        if (e->button() == Qt::LeftButton)
            mEvent = new sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::LeftReleased, e->x(), e->y());
        else if (e->button() == Qt::RightButton)
            mEvent = new sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::RightReleased, e->x(), e->y());
        else if (e->button() == Qt::MidButton)
            mEvent = new sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::MiddleReleased, e->x(), e->y());
        currentCamera->manageEvent(mEvent);

        getQWidget()->update();
    }

    void mouseEvent(QMouseEvent *e)
    {
        GLint viewport[4];
        glGetIntegerv(GL_VIEWPORT,viewport);

        MousePosition mousepos;
        mousepos.screenWidth  = viewport[2];
        mousepos.screenHeight = viewport[3];
        mousepos.x      = e->x();
        mousepos.y      = e->y();

        if (e->state() & Qt::ShiftButton)
        {

            pick.activateRay(viewport[2],viewport[3]);
            pick.updateMouse2D( mousepos );

            //_sceneTransform.ApplyInverse();
            switch (e->type())
            {
            case QEvent::MouseButtonPress:

                if (e->button() == Qt::LeftButton)
                {
                    pick.handleMouseEvent(PRESSED, LEFT);
                }
                else if (e->button() == Qt::RightButton) // Shift+Rightclick to remove triangles
                {
                    pick.handleMouseEvent(PRESSED, RIGHT);
                }
                else if (e->button() == Qt::MidButton) // Shift+Midclick (by 2 steps defining 2 input points) to cut from one point to another
                {
                    pick.handleMouseEvent(PRESSED, MIDDLE);
                }
                break;
            case QEvent::MouseButtonRelease:
                //if (e->button() == Qt::LeftButton)
            {

                if (e->button() == Qt::LeftButton)
                {
                    pick.handleMouseEvent(RELEASED, LEFT);
                }
                else if (e->button() == Qt::RightButton)
                {
                    pick.handleMouseEvent(RELEASED, RIGHT);
                }
                else if (e->button() == Qt::MidButton)
                {
                    pick.handleMouseEvent(RELEASED, MIDDLE);
                }
            }
            break;
            default:
                break;
            }
            moveRayPickInteractor(e->x(), e->y());
        }
        else
        {
            pick.activateRay(viewport[2],viewport[3]);
        }

    }

    // ---------------------- Here are the controls for instruments  ----------------------

    void moveLaparoscopic( QMouseEvent *e)
    {
        int index_instrument = simulation::getSimulation()->instrumentInUse.getValue();
        if (index_instrument < 0 || index_instrument > (int)simulation::getSimulation()->instruments.size()) return;

        simulation::Node *instrument = simulation::getSimulation()->instruments[index_instrument];
        if (instrument == NULL) return;

        int eventX = e->x();
        int eventY = e->y();

        std::vector< sofa::core::behavior::MechanicalState<sofa::defaulttype::LaparoscopicRigidTypes>* > instruments;
        instrument->getTreeObjects<sofa::core::behavior::MechanicalState<sofa::defaulttype::LaparoscopicRigidTypes>, std::vector< sofa::core::behavior::MechanicalState<sofa::defaulttype::LaparoscopicRigidTypes>* > >(&instruments);

        if (!instruments.empty())
        {
            sofa::core::behavior::MechanicalState<sofa::defaulttype::LaparoscopicRigidTypes>* instrument = instruments[0];
            switch (e->type())
            {
            case QEvent::MouseButtonPress:
                // Mouse left button is pushed
                if (e->button() == Qt::LeftButton)
                {
                    _navigationMode = BTLEFT_MODE;
                    _mouseInteractorMoving = true;
                    _mouseInteractorSavedPosX = eventX;
                    _mouseInteractorSavedPosY = eventY;
                }
                // Mouse right button is pushed
                else if (e->button() == Qt::RightButton)
                {
                    _navigationMode = BTRIGHT_MODE;
                    _mouseInteractorMoving = true;
                    _mouseInteractorSavedPosX = eventX;
                    _mouseInteractorSavedPosY = eventY;
                }
                // Mouse middle button is pushed
                else if (e->button() == Qt::MidButton)
                {
                    _navigationMode = BTMIDDLE_MODE;
                    _mouseInteractorMoving = true;
                    _mouseInteractorSavedPosX = eventX;
                    _mouseInteractorSavedPosY = eventY;
                }
                break;

            case QEvent::MouseMove:
                //
                break;

            case QEvent::MouseButtonRelease:
                // Mouse left button is released
                if (e->button() == Qt::LeftButton)
                {
                    if (_mouseInteractorMoving)
                    {
                        _mouseInteractorMoving = false;
                    }
                }
                // Mouse right button is released
                else if (e->button() == Qt::RightButton)
                {
                    if (_mouseInteractorMoving)
                    {
                        _mouseInteractorMoving = false;
                    }
                }
                // Mouse middle button is released
                else if (e->button() == Qt::MidButton)
                {
                    if (_mouseInteractorMoving)
                    {
                        _mouseInteractorMoving = false;
                        //static_cast<sofa::simulation::Node*>(instrument->getContext())->execute<sofa::simulation::GrabVisitor>();
                    }
                }
                break;

            default:
                break;
            }
            if (_mouseInteractorMoving)
            {
                {
                    helper::WriteAccessor<Data<sofa::defaulttype::LaparoscopicRigidTypes::VecCoord> > instrumentX = *instrument->write(core::VecCoordId::position());
                    if (_navigationMode == BTLEFT_MODE)
                    {
                        int dx = eventX - _mouseInteractorSavedPosX;
                        int dy = eventY - _mouseInteractorSavedPosY;
                        if (dx || dy)
                        {
                            instrumentX[0].getOrientation() = instrumentX[0].getOrientation() * Quat(Vector3(0,1,0),dx*0.001) * Quat(Vector3(0,0,1),dy*0.001);
                            _mouseInteractorSavedPosX = eventX;
                            _mouseInteractorSavedPosY = eventY;
                        }
                    }
                    else if (_navigationMode == BTMIDDLE_MODE)
                    {
                        int dx = eventX - _mouseInteractorSavedPosX;
                        int dy = eventY - _mouseInteractorSavedPosY;
                        if (dx || dy)
                        {
                            _mouseInteractorSavedPosX = eventX;
                            _mouseInteractorSavedPosY = eventY;
                        }
                    }
                    else if (_navigationMode == BTRIGHT_MODE)
                    {
                        int dx = eventX - _mouseInteractorSavedPosX;
                        int dy = eventY - _mouseInteractorSavedPosY;
                        if (dx || dy)
                        {
                            instrumentX[0].getTranslation() += (dy)*0.01;
                            instrumentX[0].getOrientation() = instrumentX[0].getOrientation() * Quat(Vector3(1,0,0),dx*0.001);
                            _mouseInteractorSavedPosX = eventX;
                            _mouseInteractorSavedPosY = eventY;
                        }
                    }
                }
                sofa::simulation::MechanicalPropagatePositionAndVelocityVisitor(core::MechanicalParams::defaultInstance()).execute(instrument->getContext());
                sofa::simulation::UpdateMappingVisitor(core::ExecParams::defaultInstance()).execute(instrument->getContext());
                //static_cast<sofa::simulation::Node*>(instrument->getContext())->execute<sofa::simulation::MechanicalPropagatePositionAndVelocityVisitor>();
                //static_cast<sofa::simulation::Node*>(instrument->getContext())->execute<sofa::simulation::UpdateMappingVisitor>();
            }
            getQWidget()->update();
        }
        else
        {
            std::vector< component::controller::Controller* > bc;
            instrument->getTreeObjects<component::controller::Controller, std::vector< component::controller::Controller* > >(&bc);

            if (!bc.empty())
            {
                switch (e->type())
                {
                case QEvent::MouseButtonPress:
                    // Mouse left button is pushed
                    if (e->button() == Qt::LeftButton)
                    {
                        sofa::core::objectmodel::MouseEvent mouseEvent(sofa::core::objectmodel::MouseEvent::LeftPressed, eventX, eventY);
                        if (groot)
                            groot->propagateEvent(&mouseEvent);
                    }
                    // Mouse right button is pushed
                    else if (e->button() == Qt::RightButton)
                    {
                        sofa::core::objectmodel::MouseEvent mouseEvent(sofa::core::objectmodel::MouseEvent::RightPressed, eventX, eventY);
                        if (groot)
                            groot->propagateEvent(&mouseEvent);
                    }
                    // Mouse middle button is pushed
                    else if (e->button() == Qt::MidButton)
                    {
                        sofa::core::objectmodel::MouseEvent mouseEvent(sofa::core::objectmodel::MouseEvent::MiddlePressed, eventX, eventY);
                        if (groot)
                            groot->propagateEvent(&mouseEvent);
                    }
                    break;

                case QEvent::MouseMove:
                {
                    if (e->state()&(Qt::LeftButton|Qt::RightButton|Qt::MidButton))
                    {
                        sofa::core::objectmodel::MouseEvent mouseEvent(sofa::core::objectmodel::MouseEvent::Move, eventX, eventY);
                        if (groot)
                            groot->propagateEvent(&mouseEvent);
                    }
                }
                break;

                case QEvent::MouseButtonRelease:
                    // Mouse left button is released
                    if (e->button() == Qt::LeftButton)
                    {
                        sofa::core::objectmodel::MouseEvent mouseEvent(sofa::core::objectmodel::MouseEvent::LeftReleased, eventX, eventY);
                        if (groot)
                            groot->propagateEvent(&mouseEvent);
                    }
                    // Mouse right button is released
                    else if (e->button() == Qt::RightButton)
                    {
                        sofa::core::objectmodel::MouseEvent mouseEvent(sofa::core::objectmodel::MouseEvent::RightReleased, eventX, eventY);
                        if (groot)
                            groot->propagateEvent(&mouseEvent);
                    }
                    // Mouse middle button is released
                    else if (e->button() == Qt::MidButton)
                    {
                        sofa::core::objectmodel::MouseEvent mouseEvent(sofa::core::objectmodel::MouseEvent::MiddleReleased, eventX, eventY);
                        if (groot)
                            groot->propagateEvent(&mouseEvent);
                    }
                    break;

                default:
                    break;
                }

                getQWidget()->update();
            }
        }
    }

    void moveLaparoscopic(QWheelEvent *e)
    {
        int index_instrument =
            simulation::getSimulation()->instrumentInUse.getValue();
        if (index_instrument < 0 || index_instrument
            > (int) simulation::getSimulation()->instruments.size())
            return;

        simulation::Node *instrument =
            simulation::getSimulation()->instruments[index_instrument];
        if (instrument == NULL)
            return;

        std::vector<component::controller::Controller*> bc;
        instrument->getTreeObjects<component::controller::Controller,
                   std::vector<component::controller::Controller*> > (&bc);

        if (!bc.empty())
        {
            sofa::core::objectmodel::MouseEvent mouseEvent(
                sofa::core::objectmodel::MouseEvent::Wheel, e->delta());
            if (groot)
                groot->propagateEvent(&mouseEvent);
            getQWidget()->update();
        }
    }

    virtual void moveRayPickInteractor(int, int)
    {
    }
    ;

    sofa::helper::gl::Capture capture;
    QTimer captureTimer;
#ifdef SOFA_HAVE_FFMPEG
    sofa::helper::gl::VideoRecorder videoRecorder;
#endif //SOFA_HAVE_FFMPEG
    sofa::simulation::Node* groot;
    std::string sceneFileName;

    bool m_isControlPressed;
    bool _video;
    bool _shadow;
    bool _gl_shadow;
    bool _axis;
    int _background;
    bool initTexturesDone;
    bool sceneBBoxIsValid;

    Vector3 backgroundColour;
    std::string backgroundImage;
    Vector3 ambientColour;

    PickHandler pick;

    //instruments handling
    int _navigationMode;
    bool _mouseInteractorMoving;
    int _mouseInteractorSavedPosX;
    int _mouseInteractorSavedPosY;

    //*************************************************************
    // QT
    //*************************************************************
    //SLOTS
    virtual void saveView()=0;
    virtual void setSizeW(int)=0;
    virtual void setSizeH(int)=0;
    virtual void captureEvent()
    {
        if (_video)
        {
            switch (SofaVideoRecorderManager::getInstance()->getRecordingType())
            {
            case SofaVideoRecorderManager::SCREENSHOTS :
                if(!captureTimer.isActive())
                    screenshot(capture.findFilename(), 1);
                break;
            case SofaVideoRecorderManager::MOVIE :
                if(captureTimer.isActive())
                {
#ifdef SOFA_HAVE_FFMPEG
                    videoRecorder.addFrame();
#endif //SOFA_HAVE_FFMPEG
                }
                break;
            default :
                break;
            }
        }
    }

    //SIGNALS
    virtual void redrawn()=0;
    virtual void resizeW(int)=0;
    virtual void resizeH(int)=0;


protected:
    sofa::component::visualmodel::BaseCamera* currentCamera;


};

class ColourPickingRenderCallBack : public sofa::gui::CallBackRender
{
public:
    ColourPickingRenderCallBack(SofaViewer* viewer):_viewer(viewer) {};
    virtual void render(core::CollisionModel::ColourCode code)
    {
        if(_viewer)
        {
            _viewer->drawColourPicking(code);
        }
    };
protected:
    SofaViewer* _viewer;

};


}
}
}
}

#endif
